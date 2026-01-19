"""
Tests for the base SapphireAPIClient.
"""

import pytest
import responses
from requests.exceptions import ConnectionError, Timeout

from sapphire_api_client import SapphireAPIClient, SapphireAPIError


class TestSapphireAPIClient:
    """Tests for base client functionality."""

    def setup_method(self):
        """Set up test client."""
        self.client = SapphireAPIClient(
            base_url="http://localhost:8000",
            max_retries=3,
            batch_size=2,  # Small batch for testing
        )

    @responses.activate
    def test_health_check_success(self):
        """Test successful health check."""
        responses.add(
            responses.GET,
            "http://localhost:8000/health",
            json={"status": "healthy"},
            status=200,
        )

        assert self.client.health_check() is True

    @responses.activate
    def test_health_check_failure(self):
        """Test health check when API is unhealthy."""
        responses.add(
            responses.GET,
            "http://localhost:8000/health",
            json={"status": "unhealthy"},
            status=200,
        )

        assert self.client.health_check() is False

    @responses.activate
    def test_readiness_check_success(self):
        """Test successful readiness check."""
        responses.add(
            responses.GET,
            "http://localhost:8000/health/ready",
            json={"status": "ready", "database": "connected"},
            status=200,
        )

        assert self.client.readiness_check() is True

    @responses.activate
    def test_readiness_check_not_ready(self):
        """Test readiness check when service is not ready."""
        responses.add(
            responses.GET,
            "http://localhost:8000/health/ready",
            status=503,
        )

        assert self.client.readiness_check() is False

    @responses.activate
    def test_get_request(self):
        """Test basic GET request."""
        responses.add(
            responses.GET,
            "http://localhost:8000/runoff/",
            json=[{"id": 1, "code": "12345"}],
            status=200,
        )

        result = self.client._get("/runoff/", params={"code": "12345"})
        assert len(result) == 1
        assert result[0]["code"] == "12345"

    @responses.activate
    def test_post_request(self):
        """Test basic POST request."""
        responses.add(
            responses.POST,
            "http://localhost:8000/runoff/",
            json=[{"id": 1, "code": "12345"}],
            status=201,
        )

        result = self.client._post("/runoff/", {"data": [{"code": "12345"}]})
        assert len(result) == 1

    @responses.activate
    def test_post_batched_single_batch(self):
        """Test batched posting with single batch."""
        responses.add(
            responses.POST,
            "http://localhost:8000/runoff/",
            json=[{"id": 1}, {"id": 2}],
            status=201,
        )

        records = [{"code": "12345"}, {"code": "12346"}]
        count = self.client._post_batched("/runoff/", records)
        assert count == 2

    @responses.activate
    def test_post_batched_multiple_batches(self):
        """Test batched posting with multiple batches."""
        # First batch
        responses.add(
            responses.POST,
            "http://localhost:8000/runoff/",
            json=[{"id": 1}, {"id": 2}],
            status=201,
        )
        # Second batch
        responses.add(
            responses.POST,
            "http://localhost:8000/runoff/",
            json=[{"id": 3}],
            status=201,
        )

        records = [{"code": "1"}, {"code": "2"}, {"code": "3"}]
        count = self.client._post_batched("/runoff/", records)
        assert count == 3

    @responses.activate
    def test_post_batched_empty_records(self):
        """Test batched posting with no records."""
        count = self.client._post_batched("/runoff/", [])
        assert count == 0

    @responses.activate
    def test_retry_on_503(self):
        """Test retry on 503 status code."""
        # First two calls return 503, third succeeds
        responses.add(
            responses.GET,
            "http://localhost:8000/test",
            status=503,
        )
        responses.add(
            responses.GET,
            "http://localhost:8000/test",
            status=503,
        )
        responses.add(
            responses.GET,
            "http://localhost:8000/test",
            json={"success": True},
            status=200,
        )

        result = self.client._get("/test")
        assert result["success"] is True
        assert len(responses.calls) == 3

    @responses.activate
    def test_fail_after_max_retries(self):
        """Test failure after max retries exhausted."""
        # All calls return 503
        for _ in range(4):  # max_retries + 1
            responses.add(
                responses.GET,
                "http://localhost:8000/test",
                status=503,
            )

        with pytest.raises(SapphireAPIError) as exc_info:
            self.client._get("/test")

        assert "Failed to connect" in str(exc_info.value)

    @responses.activate
    def test_no_retry_on_400(self):
        """Test no retry on 400 client error."""
        responses.add(
            responses.GET,
            "http://localhost:8000/test",
            json={"error": "bad request"},
            status=400,
        )

        with pytest.raises(SapphireAPIError) as exc_info:
            self.client._get("/test")

        assert exc_info.value.status_code == 400
        assert len(responses.calls) == 1  # No retries

    @responses.activate
    def test_batch_failure_stops_processing(self):
        """Test that batch failure raises error and stops."""
        # First batch succeeds
        responses.add(
            responses.POST,
            "http://localhost:8000/runoff/",
            json=[{"id": 1}, {"id": 2}],
            status=201,
        )
        # Second batch fails
        responses.add(
            responses.POST,
            "http://localhost:8000/runoff/",
            json={"error": "server error"},
            status=500,
        )

        records = [{"code": "1"}, {"code": "2"}, {"code": "3"}]

        with pytest.raises(SapphireAPIError) as exc_info:
            self.client._post_batched("/runoff/", records)

        assert "Failed at batch 2" in str(exc_info.value)


class TestAuthentication:
    """Tests for authentication functionality."""

    def test_client_without_token(self):
        """Test client without auth token."""
        client = SapphireAPIClient()
        assert client.is_authenticated is False
        assert "Authorization" not in client.session.headers

    def test_client_with_token(self):
        """Test client with auth token."""
        client = SapphireAPIClient(auth_token="my-secret-token")
        assert client.is_authenticated is True
        assert client.session.headers["Authorization"] == "Bearer my-secret-token"

    @responses.activate
    def test_token_sent_in_request(self):
        """Test that auth token is sent in request headers."""
        responses.add(
            responses.GET,
            "http://localhost:8000/test",
            json={"data": "secret"},
            status=200,
        )

        client = SapphireAPIClient(auth_token="test-token")
        client._get("/test")

        assert responses.calls[0].request.headers["Authorization"] == "Bearer test-token"

    @responses.activate
    def test_401_unauthorized_error(self):
        """Test clear error message on 401 Unauthorized."""
        responses.add(
            responses.GET,
            "http://localhost:8000/protected",
            json={"detail": "Not authenticated"},
            status=401,
        )

        client = SapphireAPIClient()

        with pytest.raises(SapphireAPIError) as exc_info:
            client._get("/protected")

        assert exc_info.value.status_code == 401
        assert "Authentication required" in str(exc_info.value)

    @responses.activate
    def test_403_forbidden_error(self):
        """Test clear error message on 403 Forbidden."""
        responses.add(
            responses.GET,
            "http://localhost:8000/admin",
            json={"detail": "Insufficient permissions"},
            status=403,
        )

        client = SapphireAPIClient(auth_token="limited-token")

        with pytest.raises(SapphireAPIError) as exc_info:
            client._get("/admin")

        assert exc_info.value.status_code == 403
        assert "Access denied" in str(exc_info.value)


class TestSapphireAPIError:
    """Tests for SapphireAPIError exception."""

    def test_error_with_status_code(self):
        """Test error with status code."""
        error = SapphireAPIError("Test error", status_code=500, response="Server error")
        assert error.status_code == 500
        assert error.response == "Server error"
        assert "Test error" in str(error)

    def test_error_without_status_code(self):
        """Test error without status code."""
        error = SapphireAPIError("Connection failed")
        assert error.status_code is None
        assert error.response is None
