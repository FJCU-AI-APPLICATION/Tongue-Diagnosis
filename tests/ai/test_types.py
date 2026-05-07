from ai.types import HeadResult, ClassScore, Normalisation, BBox


def test_class_score_holds_label_and_score():
    cs = ClassScore(label="淡紅", score=0.78)
    assert cs.label == "淡紅"
    assert cs.score == 0.78


def test_head_result_single_label():
    r = HeadResult(
        task="舌色",
        head_type="single",
        predictions=[ClassScore(label="淡紅", score=0.78)],
    )
    assert r.error is None
    assert len(r.predictions) == 1


def test_head_result_with_error_has_no_predictions():
    r = HeadResult(task="舌色", head_type="single", predictions=[], error="boom")
    assert r.error == "boom"


def test_normalisation_round_trip():
    n = Normalisation(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    assert n.mean[0] == 0.485
    assert len(n.std) == 3


def test_bbox_dataclass():
    b = BBox(x=10, y=20, w=100, h=120, confidence=0.9)
    assert b.x == 10
    assert b.confidence == 0.9
