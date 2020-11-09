import numpy as np
import copy
from PIL import Image
import os
import concurrent.futures
from sklearn.metrics import roc_curve
import json

# use below line for unit testing
test_function = '' #'pass function name here'


'''
Function to calculate integral images from given set of input images
Input: list of monochrome images
Output: list of respective integral images
'''


def get_integral_images(input_images):
    integral_image_list = []
    for idx, input_image in enumerate(input_images):
        integral_image = copy.deepcopy(input_image)
        for x in range(0, input_image.shape[0]):
            for y in range(0, input_image.shape[1]):
                if x == 0 and y == 0:
                    pass
                elif x == 0 and y != 0:
                    integral_image[x][y] = integral_image[x][y] + integral_image[x][y - 1]

                elif y == 0 and x != 0:
                    integral_image[x][y] = integral_image[x][y] + integral_image[x - 1][y]

                else:
                    integral_image[x][y] = integral_image[x][y] + integral_image[x][y - 1] + integral_image[x - 1][y] - \
                                           integral_image[x - 1][y - 1]
        integral_image_list.insert(idx, integral_image)

    return integral_image_list


'''
Function to build training data for training classifiers
Input: test image database path
Output: one list having images and other having corresponding labels
'''


def build_training_data(path_to_image_folder, count):
    training_data = []
    training_labels = []

    for subdir, dirs, files in os.walk(path_to_image_folder):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith(".ppm"):
                training_data.insert(count, np.asarray(Image.open(filepath).resize((24, 24)).convert('L')))
                training_labels.insert(count, 1)
                count += 1
            if filepath.endswith(".png"):
                training_data.insert(count, np.asarray(Image.open(filepath).resize((24, 24)).convert('L')))
                training_labels.insert(count, 0)
    return training_data, training_labels


'''
Function to build test data for testing classifiers
Input: test image database path
Output: list having face images
'''


def build_test_data(path_to_image_folder, count):
    test_data = []
    for subdir, dirs, files in os.walk(path_to_image_folder):
        for filename in files:
            filepath = subdir + os.sep + filename

            if filepath.endswith(".ppm"):
                test_data.insert(count, np.asarray(Image.open(filepath).resize((24, 24)).convert('L')))
                count += 1

    return test_data


'''
Function to calculate two vertical rectangle features
Input: list of integral images
Output: list containing extracted features
'''


def calculate_feature_1(integral_images):
    features_list = []
    for idx, integral_image in enumerate(integral_images):
        height, width = np.shape(integral_image)
        features = []
        sizeX = 1
        sizeY = 2
        for i in range(0, 24 - sizeX):
            for j in range(0, 24 - sizeY):
                wid = sizeX
                while wid <= 24 - i:
                    hei = sizeY
                    while hei <= 24 - j:
                        left_block = integral_image[hei - 1, int((wid - 1) / 2)]
                        right_block = integral_image[hei - 1, wid - 1]
                        hei += sizeY
                        features.append(np.subtract(right_block, left_block))
                    wid += sizeX
        features_list.insert(idx, features)

    return features_list


'''
Function to calculate two rectangle horizontal features
Input: list of integral images
Output: list containing extracted features
'''


def calculate_feature_2(integral_images):
    features_list = []
    for idx, integral_image in enumerate(integral_images):
        height, width = np.shape(integral_image)
        features = []
        sizeX = 2
        sizeY = 1
        for i in range(0, 24 - sizeX):
            for j in range(0, 24 - sizeY):
                wid = sizeX
                while wid <= 24 - i:
                    hei = sizeY
                    while hei <= 24 - j:
                        top_block = integral_image[int((hei - 1) / 2), wid - 1]
                        bottom_block = integral_image[hei - 1, wid - 1]
                        hei += sizeY
                        features.append(np.subtract(top_block, bottom_block))
                    wid += sizeX
        features_list.insert(idx, features)

    return features_list


'''
Function to calculate three horizontal rectangle features
Input: list of integral images
Output: list containing extracted features
'''


def calculate_feature_3(integral_images):
    features_list = []
    for idx, integral_image in enumerate(integral_images):
        height, width = np.shape(integral_image)
        features = []

        sizeX = 3
        sizeY = 1
        for i in range(0, 24 - sizeX):
            for j in range(0, 24 - sizeY):
                wid = sizeX
                while wid <= 24 - i:
                    hei = sizeY
                    while hei <= 24 - j:
                        top_block = integral_image[int((hei - 1) / 3), wid - 1]
                        middle_block = integral_image[int(((hei - 1) / 3) * 2), wid - 1]
                        bottom_block = integral_image[hei - 1, wid - 1]
                        features.append(np.subtract(middle_block, np.add(top_block, bottom_block)))

                        hei += sizeY
                    wid += sizeX
        features_list.insert(idx, features)

    return features_list


'''
Function to calculate three vertical rectangle features
Input: list of integral images
Output: list containing extracted features
'''


def calculate_feature_4(integral_images):  # 3 rectangle vertical
    features_list = []
    for idx, integral_image in enumerate(integral_images):
        height, width = np.shape(integral_image)
        features = []
        sizeX = 1
        sizeY = 3
        for i in range(0, 24 - sizeX):
            for j in range(0, 24 - sizeY):
                wid = sizeX
                while wid <= 24 - i:
                    hei = sizeY
                    while hei <= 24 - j:
                        left_block = integral_image[hei - 1, int((wid - 1) / 3)]
                        middle_block = integral_image[hei - 1, int(((wid - 1) / 3) * 2)]
                        right_block = integral_image[hei - 1, wid - 1]
                        features.append(np.subtract(middle_block, np.add(left_block, right_block)))
                        hei += sizeY
                    wid += sizeX
        features_list.insert(idx, features)

    return features_list


'''
Function to calculate four rectangle features
Input: list of integral images
Output: list containing extracted features
'''


def calculate_feature_5(integral_images):  # four rectangle feature
    features_list = []
    for idx, integral_image in enumerate(integral_images):
        height, width = np.shape(integral_image)
        features = []
        sizeX = 2
        sizeY = 2
        for i in range(0, 24 - sizeX):
            for j in range(0, 24 - sizeY):
                wid = sizeX
                while wid <= 24 - i:
                    hei = sizeY
                    while hei <= 24 - j:
                        top_left = integral_image[int((wid - 1) / 2), int((hei - 1) / 2)]
                        top_right = integral_image[wid - 1, int((hei - 1) / 2)]
                        bottom_left = integral_image[int((wid - 1) / 2), hei - 1]
                        bottom_right = integral_image[wid - 1, hei - 1]
                        features.append(np.subtract(np.add(top_right, bottom_left), np.add(top_left, bottom_right)))
                        hei += sizeY
                    wid += sizeX
        features_list.insert(idx, features)

    return features_list


'''
AdaBoost
The function to calculate n best features among the input features
Input: list of features, length of positive dataset, length of negative dataset
Output: list of best features 
'''


def ada_boost(itrs, feature_data, length_positive, length_negative, training_labels, weight_matrix):
    list_of_best_features = []
    list_of_alpha = []
    for T in range(0, itrs):
        # Normalize weight matrix
        weight_matrix = np.divide(weight_matrix, np.sum(weight_matrix))
        feature_vs_img_matrix = weight_matrix.transpose() * feature_data
        fpr, tpr, estimated_threshold = roc_curve(training_labels, weight_matrix[0])

        error_matrix = copy.deepcopy(feature_vs_img_matrix)
        error_matrix[error_matrix >= estimated_threshold[0]] = 1  # 1 for positive
        error_matrix[error_matrix < estimated_threshold[0]] = 0  # 0 for negative

        pos_mat = error_matrix[:length_positive, :]
        neg_mat = error_matrix[length_positive:, :]
        temp_weight_matrix = copy.deepcopy(weight_matrix)

        pos_mat_error = np.subtract(np.sum(weight_matrix),
                                    temp_weight_matrix[:, :length_positive].transpose() * pos_mat)
        pos_mat_error_count = np.sum(pos_mat_error, axis=0)
        temp_weight_matrix1 = copy.deepcopy(weight_matrix)
        neg_mat_error = temp_weight_matrix1[:, length_positive:].transpose() * neg_mat
        neg_mat_error_count = np.sum(neg_mat_error, axis=0)
        total_error_count = np.add(pos_mat_error_count, neg_mat_error_count)
        best_feature_index = np.argmin(total_error_count)

        list_of_best_features.insert(T, int(best_feature_index))
        beta_t = np.divide(total_error_count[best_feature_index], 1 - total_error_count[best_feature_index])
        alpha_t = np.log(1 / np.absolute(beta_t))
        list_of_alpha.insert(T, alpha_t)

        ###update weights
        pos_mat = error_matrix[:length_positive, best_feature_index]
        neg_mat = error_matrix[length_positive:, best_feature_index]

        neg_mat_1 = neg_mat

        # Use this to restore weight in the last step
        pos_mat_1 = pos_mat + 1
        pos_mat_1 = np.where(pos_mat_1 == 2, 0, pos_mat_1)

        neg_mat = neg_mat + 1
        neg_mat = np.where(neg_mat == 2, 0, neg_mat)

        pos_updated_wt_mat = weight_matrix[0][:length_positive]
        neg_updated_wt_mat = weight_matrix[0][length_positive:]

        # for all correctly classified examples multiply with beta
        decrease_pos_wt = (pos_updated_wt_mat * pos_mat) * (
            np.divide(total_error_count[best_feature_index], 1 - total_error_count[best_feature_index]))
        decrease_neg_wt = (neg_updated_wt_mat * neg_mat) * (
            np.divide(total_error_count[best_feature_index], 1 - total_error_count[best_feature_index]))

        dummy_weight_mat = np.concatenate((decrease_pos_wt, decrease_neg_wt), axis=None)

        restore_weight = np.concatenate((pos_mat_1, neg_mat_1), axis=None)

        restore_weight_1 = weight_matrix * restore_weight

        # updated weight matrix
        weight_matrix = dummy_weight_mat + restore_weight_1

    return list_of_best_features, list_of_alpha, weight_matrix


'''
Function to generate cascade blocks which contain set of weak classifiers generated by AdaBoost
'''


def build_cascades(test_data, list_of_test_data_features, list_of_all_features, length_positive, length_negative,
                   training_labels):
    true_positive_rate = 0.0
    cascade_block_1 = []
    cascade_block_2 = []
    cascade_block_3 = []
    cascade_block_4 = []
    cascade_1_alpha = []
    cascade_2_alpha = []
    cascade_3_alpha = []
    cascade_4_alpha = []
    positive_data_size = len(test_data)
    compute_features = 0.0
    positive_count = 0
    w1 = 1 / (2 * length_positive)
    w2 = 1 / (2 * length_negative)

    positive_weight_matrix = [w1] * length_positive
    negative_weight_matrix = [w2] * length_negative

    weight_matrix = positive_weight_matrix + negative_weight_matrix

    weight_matrix = np.asarray(weight_matrix)
    weight_matrix = [weight_matrix]
    weight_matrix = np.asarray(weight_matrix)

    while true_positive_rate <= 0.5:
        best_features_list, alpha_values, weight_matrix = ada_boost(200, list_of_all_features, length_positive,
                                                                    length_negative, training_labels, weight_matrix)
        cascade_block_1.extend(best_features_list)
        cascade_1_alpha.extend(alpha_values)
        for j in list_of_test_data_features:
            for i, feature_index in enumerate(cascade_block_1):
                compute_features += j[feature_index] * alpha_values[i]
            if compute_features >= 0.5 * np.sum(alpha_values):
                positive_count += 1

        true_positive_rate = positive_count / positive_data_size

    true_positive_rate = 0
    positive_count = 0

    while true_positive_rate <= 0.6:
        best_features_list, alpha_values, weight_matrix = ada_boost(200, list_of_all_features, length_positive,
                                                                    length_negative,
                                                                    training_labels, weight_matrix)
        cascade_block_2.extend(best_features_list)
        cascade_2_alpha.extend(alpha_values)

        for j in list_of_test_data_features:
            for i, feature_index in enumerate(cascade_block_2):
                compute_features += j[feature_index] * alpha_values[i]
            if compute_features >= 0.5 * np.sum(alpha_values):
                positive_count += 1

        true_positive_rate = positive_count / positive_data_size

    true_positive_rate = 0
    positive_count = 0
    while (true_positive_rate <= 0.7):
        itr = 0
        best_features_list, alpha_values, weight_matrix = ada_boost(200, list_of_all_features, length_positive,
                                                                    length_negative,
                                                                    training_labels, weight_matrix)

        cascade_block_3.extend(best_features_list)
        cascade_3_alpha.extend(alpha_values)

        for j in list_of_test_data_features:
            for i, feature_index in enumerate(cascade_block_3):
                compute_features += j[feature_index] * alpha_values[i]
            if compute_features >= 0.5 * np.sum(alpha_values):
                positive_count += 1

        true_positive_rate = positive_count / positive_data_size

    true_positive_rate = 0
    positive_count = 0

    while (true_positive_rate <= 0.8):

        best_features_list, alpha_values, weight_matrix = ada_boost(200, list_of_all_features, length_positive,
                                                                    length_negative,
                                                                    training_labels, weight_matrix)
        cascade_block_4.extend(best_features_list)
        cascade_4_alpha.extend(alpha_values)

        for j in list_of_test_data_features:
            for i, feature_index in enumerate(cascade_block_4):
                compute_features += j[feature_index] * alpha_values[i]
            if compute_features >= 0.5 * np.sum(alpha_values):
                positive_count += 1

        true_positive_rate = positive_count / positive_data_size

    list_of_cascades = [cascade_block_1, cascade_block_2, cascade_block_3, cascade_block_4]
    list_of_alpha = [cascade_1_alpha, cascade_2_alpha, cascade_3_alpha, cascade_4_alpha]
    return list_of_cascades, list_of_alpha


'''
Controller function
'''


def main():
    training_data, training_labels = build_training_data('faceData/trainData', 0)
    test_data = build_test_data('faceData/testData', 0)

    training_data = np.array(training_data)
    test_data = np.array(test_data)

    training_labels = np.array(training_labels)
    inds = training_labels.argsort()
    training_data = training_data[inds]

    length_positive = np.count_nonzero(training_labels)
    length_negative = len(training_data) - length_positive

    integral_images = get_integral_images(training_data)
    test_data_integral_images = get_integral_images(test_data)

    # use concurrent futures for parallel processing of the data
    with concurrent.futures.ThreadPoolExecutor() as executor:
        feature1_data = executor.submit(calculate_feature_1, integral_images)
        feature2_data = executor.submit(calculate_feature_2, integral_images)
        feature3_data = executor.submit(calculate_feature_3, integral_images)
        feature4_data = executor.submit(calculate_feature_4, integral_images)
        feature5_data = executor.submit(calculate_feature_5, integral_images)
        test_feature1_data = executor.submit(calculate_feature_1, test_data_integral_images)
        test_feature2_data = executor.submit(calculate_feature_2, test_data_integral_images)
        test_feature3_data = executor.submit(calculate_feature_3, test_data_integral_images)
        test_feature4_data = executor.submit(calculate_feature_4, test_data_integral_images)
        test_feature5_data = executor.submit(calculate_feature_5, test_data_integral_images)
        feature1 = feature1_data.result()
        feature2 = feature2_data.result()
        feature3 = feature3_data.result()
        feature4 = feature4_data.result()
        feature5 = feature5_data.result()
        t_feature1 = test_feature1_data.result()
        t_feature2 = test_feature2_data.result()
        t_feature3 = test_feature3_data.result()
        t_feature4 = test_feature4_data.result()
        t_feature5 = test_feature5_data.result()

    list_of_all_features = copy.deepcopy(feature1)
    list_of_all_features = np.append(list_of_all_features, feature2, 1)
    list_of_all_features = np.append(list_of_all_features, feature3, 1)
    list_of_all_features = np.append(list_of_all_features, feature4, 1)
    list_of_all_features = np.append(list_of_all_features, feature5, 1)
    list_of_all_features = np.asarray(list_of_all_features)

    list_of_test_data_features = copy.deepcopy(t_feature1)
    list_of_test_data_features = np.append(list_of_test_data_features, t_feature2, 1)
    list_of_test_data_features = np.append(list_of_test_data_features, t_feature3, 1)
    list_of_test_data_features = np.append(list_of_test_data_features, t_feature4, 1)
    list_of_test_data_features = np.append(list_of_test_data_features, t_feature5, 1)

    list_of_cascades, list_of_alpha = build_cascades(test_data, list_of_test_data_features, list_of_all_features,
                                                     length_positive,
                                                     length_negative, training_labels)

    # export model.json
    json_arr = {}
    json_arr["Cascade1"] = list_of_cascades[0]
    json_arr["Cascade2"] = list_of_cascades[1]
    json_arr["Cascade3"] = list_of_cascades[2]
    json_arr["Cascade4"] = list_of_cascades[3]
    json_arr["Alpha1"] = list_of_alpha[0]
    json_arr["Alpha2"] = list_of_alpha[1]
    json_arr["Alpha3"] = list_of_alpha[2]
    json_arr["Alpha4"] = list_of_alpha[3]

    json_string = json.dumps(json_arr)

    with open('model.json', 'w') as json_file:
        json.dump(json_string, json_file)


if __name__ == "__main__":
    main()

############################  test functions  ###################

if test_function == 'get_integral_image':
    # test get integral image
    # test = np.zeros(25).reshape(5,5)
    # test = [[1, 7, 4, 2, 9], [7, 2, 3, 8, 2], [1, 8, 7, 9, 1], [3, 2, 3, 1, 5], [2, 9, 5, 6, 6]]
    master_list = []
    test = np.arange(16).reshape(4, 4)
    test1 = np.arange(16).reshape(4, 4)
    master_list.insert(0, test)
    master_list.insert(1, test1)
    # test = np.append(test, test1, 0)
    print(master_list)
    print('')
    print(get_integral_images(master_list))

if test_function == 'build_training_data':
    count, label = build_training_data('faceData', 0)
    print(count[0].shape)
    print(len(count))
    print(len(label))

if test_function == 'calculate_feature_1':
    # test = [[1, 7, 4, 2], [7, 2, 3, 8], [1, 8, 7, 9], [3, 2, 3, 1]]
    master_list = []
    test = np.arange(576).reshape(24, 24)
    # test = np.append(test,test1,0)

    test = np.asarray(test)
    master_list.insert(0, test)
    np.asarray(master_list)
    test_input = get_integral_images(master_list)
    # print(test_input)
    print(len(calculate_feature_1(test_input)[0]))

if test_function == 'calculate_feature_2':
    test = [[1, 7, 4, 2], [7, 2, 3, 8], [1, 8, 7, 9], [3, 2, 3, 1]]
    test = np.asarray(test)
    test_input = get_integral_images(test)
    print(test_input)
    print(calculate_feature_2(test_input, 0))

if test_function == 'calculate_feature_3':
    test = [[1, 7, 4, 2], [7, 2, 3, 8], [1, 8, 7, 9], [1, 7, 4, 2], [7, 2, 3, 8], [1, 8, 7, 9]]
    test = np.asarray(test)
    test_input = get_integral_images(test)
    print(test_input)
    print(calculate_feature_3(test_input, 0))

if test_function == 'calculate_feature_4':
    test = [[1, 7, 4], [7, 2, 3], [1, 8, 7], [1, 7, 4], [7, 2, 3], [1, 8, 7]]
    test = np.asarray(test)
    test_input = get_integral_images(test)
    print(test_input)
    print(calculate_feature_4(test_input, 0))

if test_function == 'calculate_feature_5':
    test = [[1, 7, 4, 2], [7, 2, 3, 8], [1, 8, 7, 9], [1, 7, 4, 2]]
    test = np.asarray(test)
    test_input = get_integral_images(test)
    print(test_input)
    print(calculate_feature_5(test_input, 0))
