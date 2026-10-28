from services.color_engine.skin_model import CLASSES, predict_season


def _features(L, a, b, chroma, hue, ita):
    return {"skin_lab": [L, a, b], "chroma": chroma, "hue_angle": hue, "ita": ita}


def test_returns_all_four_seasons_ranked_and_normalized():
    ranked = predict_season(_features(70.77, 16.35, 28.14, 32.55, 59.85, 36.43))
    assert sorted(s for s, _ in ranked) == sorted(CLASSES)
    probs = [p for _, p in ranked]
    assert probs == sorted(probs, reverse=True)
    assert abs(sum(probs) - 1.0) < 1e-3


def test_matches_training_output_on_a_known_photo():
    # Deep Armocromia image 10306 (truth Autumn); the sklearn model trained by
    # personal_color_ai/skin_model/train_skin_model.py ranks autumn first.
    assert predict_season(_features(70.7702, 16.3506, 28.1444, 32.5492, 59.8454, 36.4268))[0][0] == "autumn"
