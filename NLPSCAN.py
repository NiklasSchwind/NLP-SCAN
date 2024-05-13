
from NLPSCAN_pipeline import NLPSCAN



embedding_model = 'IndicativeSentence'  #IndicativeSentence or SBert
indicative_sentence = 'Category: <mask>. ' #Sentence with a class indicative word masked out
indicative_sentence_position = 'first' #Position of your class indicative sentence in the text, first or last
num_classes = 1 # Number of classes in your dataset
file_path_data = ''  #Path to your data
file_path_result = '' # Path to save your results
device = 'cpu' # cuda:n for nth cuda or cuda if only one is available




if __name__ == "__main__":
    NLPSCAN = NLPSCAN(file_path_data = file_path_data,
                    file_path_result=file_path_result,
                    embedding_method = embedding_model,
                    num_classes = num_classes,
                    indicative_sentence = indicative_sentence,
                    indicative_sentence_position= indicative_sentence_position,
                    device = device)

    NLPSCAN.TrainAndClassify()


