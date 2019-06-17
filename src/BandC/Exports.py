from pandas.core.frame import DataFrame
import urllib.parse
import arff


def export_csv(data: DataFrame):
    dataset_string = data.to_csv(index=False, encoding='utf-8')
    return "data:text/csv;charset=utf-8," + \
           urllib.parse.quote(dataset_string)

def export_arff(file_name: str, data: DataFrame, attributes, description: str):
    if description is None:
        description = ''
    if attributes is None:
        attributes = [(attribute, "STRING") for attribute in data.columns]
    arff_dict = {'description': description,
                 'relation': file_name,
                 'attributes': attributes,
                 'data': data.values.tolist()}
    return "data:text/arff;charset=utf-8," + \
           urllib.parse.quote(arff.dumps(arff_dict))
