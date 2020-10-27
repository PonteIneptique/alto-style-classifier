from stylalto.tagger import Tagger
import glob

tagger = Tagger.load_from_prefix("example_model/model2")
data = tagger.tag(glob.glob("test_data/*.xml"), batch_size=4)
print(data)