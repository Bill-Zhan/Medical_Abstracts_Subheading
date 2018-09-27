
def SentenceLabel_wrapper(single_sentence:str,model_name:str,train_file:str
                          worddict_json:str,chardict_json:str,maxlen_txt:str):
    '''
    Input: sentence to label, pretrained model
    Ouput: predicted sentence label
    :param single_sentence: string, sentence to label
    :param model_name: pretrained keras model file name
    :param train_file: train csv file name
    :param worddict_json: saved word dict json file name
    :param chardict_json: saved char dict json file name
    :maxlen_txt: max word length file name
    :return a label: predicted label
    '''
    import json
    import numpy as np
    import pandas as pd
    from keras.models import load_model
    from sklearn.preprocessing import LabelEncoder
    from keras_wc_embd import get_word_list_eng, get_dicts_generator, get_batch_input

    #--- read train_data
    train = pd.read_csv(train_file)
    sentences_train = train['SENTENCE'].fillna("DUMMY_VALUE").values  #numpy array
    le = LabelEncoder()
    targets_train = train['LABEL'].values
    le.fit(targets_train)

    #--- load model
    model = load_model(model_name)
    test_num = 1
    test_steps = 1
    print('\nmodel loaded')
    
    #--- load word_dict, char_dict and word_len
    with open(worddict_json) as json_wd:
        word_dict = json.loads(json_wd.read()) 
    with open(chardict_json) as json_cd:
        char_dict = json.loads(json_cd.read())
    with open(maxlen_txt) as f:
        max_word_len = int(f.read())
    print('word and char dicts loaded')
    
    #--- test function
    def test_batch_generator(batch_size=1):
        index = 0
        while index < test_num:
            sentences = []
            index += 1
            sentences.append(get_word_list_eng(single_sentence))
            word_input, char_input = get_batch_input(
                sentences=sentences,
                max_word_len=max_word_len,
                word_dict=word_dict,
                char_dict=char_dict,
                word_ignore_case=True,
                char_ignore_case=False,
            )
            yield [word_input, char_input]

    predicts = model.predict_generator(
        generator=test_batch_generator(),
        steps=test_steps,
        verbose=True,
    )
    predicts = np.argmax(predicts, axis=-1).tolist()

    pred_label = le.inverse_transform(predicts[0])

    return pred_label