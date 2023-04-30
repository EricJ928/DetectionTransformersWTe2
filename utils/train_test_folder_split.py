import splitfolders

input_folder = r'WTe2_recognition/WTe2_examples/all/png/good/object_detection/WTe2'
output_folder = r'WTe2_recognition/WTe2_examples/all/png/good/object_detection/WTe2'

splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(0.8, 0.1, 0.1))
