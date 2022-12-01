from rawsight.datasets import load_course2_week3_data

dataset = load_course2_week3_data()
print(dataset.X_train.shape)
print(dataset.X_cv.shape)
print(dataset.X_test.shape)
