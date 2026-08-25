import json

from driftguard.training.vspw_segformer import load_vspw_class_names


def test_class_order_excludes_raw_others_and_maps_one_to_zero(tmp_path):
    mapping = {"others": "0"}
    mapping.update({f"class_{index}": str(index) for index in range(1, 125)})
    (tmp_path / "label_num_dic_final.json").write_text(json.dumps(mapping), encoding="utf-8")
    names = load_vspw_class_names(tmp_path)
    assert len(names) == 124
    assert names[0] == "class_1"
    assert names[123] == "class_124"
