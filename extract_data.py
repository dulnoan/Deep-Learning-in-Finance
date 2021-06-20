import re
import os
import argparse
import datetime
import pathlib

import pdftotext
import pandas as pd

columns = ['Date', 'Company Ticker', 'Buy/Sell', 'Summary','Summary_processed','Body','Body_processed','Left Column Data','File']

def get_pageData(file_path):
    first_column_end = 81
    with open(file_path, "rb") as f:
        pdf = pdftotext.PDF(f,raw=False)
    first_page = pdf[0]
    lines=first_page.splitlines()


    data_obj =dict.fromkeys(columns)
    data_col = []
    content_col = []
    start_data = False
    for idx,line in enumerate(lines[:-3]):
        date_ptrn = re.compile('\d{1,2} [A-Z]{1}[a-z]{2,10} \d{4}')
        match = date_ptrn.findall(line[:first_column_end])
        if match:
            start_data = True
            date_str = line[:first_column_end-10].strip()
            prev_line = lines[idx-1]

            quote = line[len(date_str)+2:].strip()
            name = prev_line[len(date_str)+5:].strip()
            first_column_end = prev_line.find(name)-2
            if not quote:
                quote = name.replace("#","")
            data_obj[columns[0]] = datetime.datetime.strptime(date_str, '%d %B %Y').date().strftime('%d/%m/%Y')
            data_obj[columns[1]] = quote
            data_obj[columns[2]] = prev_line[:first_column_end].strip()

        if start_data:
            data_col.append((line[:first_column_end+1]))
            content_col.append(line[first_column_end:])

    splitted_content =  "\n".join(content_col).split("\n\n\n\n\n")
    data_obj[columns[3]] = splitted_content[0]
    data_obj[columns[4]] = splitted_content[0].strip().replace("\n","").lower()

    if len(splitted_content) > 1:
        data_obj[columns[5]] = "".join(splitted_content[1:])
        data_obj[columns[6]] = "".join(splitted_content[1:]).strip().replace("\n","").lower()


    data_obj[columns[7]] = "\n".join(data_col[1:])
    data_obj[columns[-1]] = file_path.name
    return data_obj

def main(load_path):
    assert os.path.exists(load_path), 'The path {} does not exist'.format(load_path)

    path_list = [path for path in load_path.iterdir()]
    all_rows = []
    for filepath in path_list:
        filename = os.path.basename(filepath)
        try:
            row = get_pageData(filepath)
            all_rows.append(row)
        except:
            print('Some problem occured in file {} skipping'.format(filename))
            row = None
            continue

        print('Processing complete for file {}'.format(filename))

    df = pd.DataFrame.from_dict(all_rows)
    df.to_excel('output.xlsx',index=False)

    print("Where this was loaded from ",load_path)

    print('\nProcessing complete for all files. Check output.xlsx file.')

    return True




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Pdf data extraction parameters')

    parser.add_argument('-i',
            '--input_path',
            default = './input',
            help = 'Path to pdf data file or directory')

    arguments = parser.parse_args()
    data_path = pathlib.Path(arguments.input_path)
    main(data_path)
    
