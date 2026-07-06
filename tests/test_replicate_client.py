from unittest.mock import patch

from modelz import replicate_client


def test_run_model_opens_path_inputs_and_calls_replicate_run(tmp_path):
    file_input = tmp_path / "in.mp4"
    file_input.write_bytes(b"fake video bytes")

    with patch("modelz.replicate_client.replicate.run") as mock_run:
        mock_run.return_value = "https://replicate.delivery/out.mp4"

        result = replicate_client.run_model(
            "owner/model", {"video": file_input, "seed": 42}
        )

    assert result == "https://replicate.delivery/out.mp4"
    called_input = mock_run.call_args.kwargs["input"]
    assert called_input["seed"] == 42
    assert called_input["video"].name == str(file_input)


def test_run_model_unwraps_list_output(tmp_path):
    file_input = tmp_path / "in.mp4"
    file_input.write_bytes(b"fake")

    with patch("modelz.replicate_client.replicate.run") as mock_run:
        mock_run.return_value = ["https://replicate.delivery/out.mp4"]

        result = replicate_client.run_model("owner/model", {"video": file_input})

    assert result == "https://replicate.delivery/out.mp4"


def test_download_saves_url_to_dest(tmp_path):
    dest = tmp_path / "out" / "final.mp4"

    with patch("modelz.replicate_client.urllib.request.urlretrieve") as mock_urlretrieve:
        result = replicate_client.download("https://replicate.delivery/out.mp4", dest)

    assert result == dest
    assert dest.parent.exists()
    mock_urlretrieve.assert_called_once_with(
        "https://replicate.delivery/out.mp4", str(dest)
    )
