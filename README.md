Stylalto
========

1. Install requirements.txt
2. Put your alto in folders such as `"./input/**/*.xml"` finds it
3. Run `01_run_extract.py`
4. Run `02_train.py`

You can also use `python3 -m stylalto.cli` as a Command Line Interface

## ToDo

- [ ] Interrupt after n non-improvement
- [x] Save classes with model as well as model name
- [x] Prediction on ALTO
  - [x] Load models
  - [x] Prediction
  - [x] Write
- [ ] Try other models
- [ ] Extract, train, tag and test to be able to use binarize ( See https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html)