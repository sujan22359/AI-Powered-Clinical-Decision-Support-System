# Test Fixtures

This directory contains test fixtures for the Clinical Decision Support System test suite.

## Files

### Clinical Text Files
- `sample_clinical_text.txt` - Clinical report with abnormal values (high BP, glucose, etc.)
- `sample_clinical_text_normal.txt` - Clinical report with all normal values
- `sample_clinical_text_mixed.txt` - Clinical report with mixed normal/borderline values

### Image Files
- `sample_xray.jpg` - Test chest X-ray image (JPEG format)
- `sample_ct_brain.jpg` - Test CT brain scan image (JPEG format)
- `sample_image.png` - Test medical image (PNG format)
- `corrupted_image.jpg` - Corrupted file for error testing

## Regenerating Fixtures

To regenerate the image fixtures, run:
```bash
python backend/tests/create_test_fixtures.py
```

## Usage

These fixtures are automatically loaded by pytest through the conftest.py fixtures.
Use them in tests like:

```python
def test_something(sample_clinical_text):
    # sample_clinical_text is automatically loaded
    assert "Blood Pressure" in sample_clinical_text
```
