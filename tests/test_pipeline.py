import sys
import json
import py_compile
from pathlib import Path

def test_pipeline():
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    
    # 1. Compile Python scripts
    py_compile.compile(str(repo_root / "scripts" / "download_data.py"), doraise=True)
    py_compile.compile(str(repo_root / "scripts" / "convert_manga109_to_yolo.py"), doraise=True)

    # 2. Check Jupyter Notebook JSON
    with open(repo_root / "notebooks" / "train_yolov11.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 5

    # 3. Unit test conversion logic
    from scripts.convert_manga109_to_yolo import convert_to_yolo_format, parse_xml

    cls_id, xc, yc, w, h = convert_to_yolo_format(100, 200, 300, 400, 1000, 1000, class_id=0)
    assert cls_id == 0
    assert abs(xc - 0.2) < 1e-5
    assert abs(yc - 0.3) < 1e-5
    assert abs(w - 0.2) < 1e-5
    assert abs(h - 0.2) < 1e-5

    # 4. XML Parsing
    sample_xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<book title="TestBook">\n'
        '  <pages>\n'
        '    <page index="0" width="1000" height="2000">\n'
        '      <frame id="0" xmin="50" ymin="50" xmax="950" ymax="1950"/>\n'
        '      <face id="1" xmin="100" ymin="100" xmax="300" ymax="300"/>\n'
        '      <body id="2" xmin="100" ymin="100" xmax="500" ymax="800"/>\n'
        '      <text id="3" xmin="200" ymin="400" xmax="400" ymax="800">Hello</text>\n'
        '      <text id="4" xmin="600" ymin="1000" xmax="800" ymax="1400">Speech</text>\n'
        '    </page>\n'
        '  </pages>\n'
        '</book>\n'
    )
    test_file = repo_root / "tests" / "temp_sample.xml"
    test_file.write_text(sample_xml, encoding="utf-8")
    try:
        parsed = parse_xml(test_file)
        assert len(parsed) == 1
        assert parsed[0]["page_index"] == 0
        assert len(parsed[0]["texts"]) == 2
    finally:
        if test_file.exists():
            test_file.unlink()

    # 5. Config YAML check
    yaml_file = repo_root / "manga_text.yaml"
    assert yaml_file.exists()
    content = yaml_file.read_text(encoding="utf-8")
    assert "nc: 1" in content
    assert "'text'" in content

    print("ALL TESTS PASSED!")

if __name__ == "__main__":
    test_pipeline()
