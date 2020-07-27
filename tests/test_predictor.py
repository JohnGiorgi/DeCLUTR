class TestDeCLUTRPredictor:
    def test_json_to_instance(self, predictor) -> None:
        json_dict = {"text": "They may take our lives, but they'll never take our freedom!"}
        output = predictor._json_to_instance(json_dict)
        assert "anchors" in output
        assert "positives" not in output
