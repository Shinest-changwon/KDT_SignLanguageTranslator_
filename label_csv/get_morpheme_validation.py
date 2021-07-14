import os
import json
import csv

basic_path = os.getcwd()

first_files = os.listdir(basic_path)
first_folders = [folder for folder in first_files if os.path.isdir(basic_path+'/'+folder)] # '[라벨링]01_crowd_morpheme' etc...

json_file_path = []
for first_folder in first_folders:
    second_folders = os.listdir(basic_path + '/' + first_folder + '/morpheme') # '01' etc...
    for second_folder in second_folders:
        # print('first_folder : {}, second_folder : {}'.format(first_folder,second_folder))
        json_file_path.append(basic_path + '/' + first_folder + '/morpheme/' + second_folder) # 총 6개 폴더
print(len(json_file_path)) # 6개(폴더)

# csv로 저장
with open('morpheme_validation.csv', 'w', newline='',encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['filename', 'meaning','attribute'])
 
for path in json_file_path:
    json_file_list = []
    # print('json_file_list len: ',len(json_file_list))
    for json_file_path in os.listdir(path):
        json_file_list.append(path + '/' +json_file_path)
    # print('json_file_list 0:3: ',json_file_list[:3])
    for json_path in json_file_list:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            with open('morpheme_validation.csv', 'a', newline='',encoding='utf-8') as csv_file:
                print('{} 시도'.format(json_path))
                writer = csv.writer(csv_file)
                file_name = json_data['metaData']['name']
                
                if len(json_data['data']) > 0:
                    for i in range(len(json_data['data'])):
                        meaning = json_data['data'][i]['attributes'][0]['name']
                        try:
                            attribute = json_data['data'][i]['attributes'][0]['attribute']
                        except:
                            attribute = ''
                        writer.writerow([file_name,meaning,attribute])
                else:
                    writer.writerow([file_name,'',''])