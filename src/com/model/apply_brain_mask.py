import scipy.io as scio
import os
import numpy as np

def load_brain_alice():
    import nibabel as nib
    participants = {'18', '22', '23', '24', '26', '28', '30', '31', '35', '36', '37',
                    '39', '41', '42', '43', '44', '45', '47', '48', '49', '50', '51', '53'}
    ROI_path = '../../../resource/roi.mat'

    data = scio.loadmat(ROI_path)
    # read roi index
    roi = data['index']
    for user in participants:
        func_filename = '/Storage/ying/resources/AliceDataset/fMRI/sub-' + str(user) + \
                        '/derivatives/sub-' + str(user) + '_task-alice_bold_preprocessed.nii.gz'
        nii = nib.load(func_filename)
        img_np = nii.get_fdata().T
        img_np = img_np.reshape(img_np.shape[0],-1)

        for i in range(10,img_np.shape[0]):
            save_path = '/Storage/ying/resources/BrainBertTorch/brain/alice/ae_npy/' + \
                             'alice_' + str(user)  + '_' + str(i-10) + '.npy'
            a = []
            print('alice_' + str(user)  + '_' + str(i-10) + '.npy')
            for index in roi[0]:
                a.append(img_np[i][index])
            np.save(save_path, a)




def load_brain_pereira():
    paticipants_1 = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07',
                     'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17', 'P01']
    paticipants_2 = ['M02', 'M04', 'M07', 'M08', 'M09', 'M14', 'M15', 'P01']
    paticipants_3 = ['M02', 'M03', 'M04', 'M07', 'M15', 'P01']
    # dataset_path = "/Storage/ying/resources/pereira2018/'+user+'/data_180concepts_wordclouds.mat"
    for user in paticipants_1:
        # 用于训练AE
        exp1_path = '/Storage/ying/resources/pereira2018/' + user + '/data_180concepts_sentences.mat'
        # 其他提取特征量
        # exp1_path = '/Storage/ying/resources/pereira2018/' + user + '/data_180concepts_wordclouds.mat'

        exp1_data = scio.loadmat(exp1_path)
        examples = exp1_data['examples']
        print(user, examples.shape)
        # ROI_path = '../../../resource/' + user + '_roi.mat'
        ROI_path = '../../../resource/'+user+'_roi.mat'

        data = scio.loadmat(ROI_path)
        # read roi index
        roi = data['index']
        for i in range((examples.shape[0])):
            exp1_save_path = '/Storage/ying/resources/BrainBertTorch/brain/pereira/npy/AE_training/' + \
                             'pereira_' + user + '_exp1_' + str(i) + '.npy'
            a = []
            for index in roi[0]:
                # roi 文件的坐标值 需要在
                if index < len(examples[i]):
                    a.append(examples[i][index])
                # else:
                    # print(i,index)
            np.save(exp1_save_path, a)

        print(len(a))
        # if user in paticipants_2:
        #     exp2_path = '/Storage/ying/resources/pereira2018/' + user + '/data_384sentences.mat'
        #     exp2_data = scio.loadmat(exp2_path)
        #     # print(exp2_data.keys())
        #     examples = exp2_data['examples_passagesentences']
        #     for i in range((examples.shape[0])):
        #         exp2_save_path = '/Storage/ying/resources/BrainBertTorch/brain/pereira/npy/' + \
        #                          'pereira_' + user + '_exp2_' + str(i) + '.npy'
        #         # print(exp2_save_path)
        #         a = []
        #         for index in roi[0]:
        #             a.append(examples[i][index])
        #         # print(len(a))
        #         np.save(exp2_save_path, a)
        #
        # if user in paticipants_3:
        #     exp3_path = '/Storage/ying/resources/pereira2018/' + user + '/data_243sentences.mat'
        #     exp3_data = scio.loadmat(exp3_path)
        #     # print(exp3_data.keys())
        #     examples = exp3_data['examples_passagesentences']
        #     for i in range((examples.shape[0])):
        #         print(i)
        #         exp3_save_path = '/Storage/ying/resources/BrainBertTorch/brain/pereira/npy/' + \
        #                          'pereira_' + user + '_exp3_' + str(i) + '.npy'
        #         # print(exp3_save_path)
        #         a = []
        #         for index in roi[0]:
        #             a.append(examples[i][index])
        #         # print(len(a))
        #         np.save(exp3_save_path, a)

# pereira
# paticipants_1 = ['M09']


# alice


# save the new file for brain


if __name__ == "__main__":
    load_brain_pereira()
    # ROI_path = '../../../resource/roi.mat'
    #
    # data = scio.loadmat(ROI_path)
    # # read roi index
    # roi = data['index']
    # print(roi)