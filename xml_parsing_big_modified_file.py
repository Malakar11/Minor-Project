from xml.dom import minidom
import csv
import pandas as pd
import numpy as np
def Make_Dataframe(file_name,count=0):
    details={'PM2.5':'','PM10':'','NO2':'','NH3':'','SO2':'','CO':'','OZONE':'','Air Quality Index':''}
    data_list = []
    data_tuple_list = []
    missing_field_flag = False
    xmldoc = minidom.parse(file_name)
    AqIndex= xmldoc.getElementsByTagName("AqIndex")[0]
    Country = AqIndex.getElementsByTagName("Country")[0]
    #print("Country:",Country.getAttribute("id"))
    State = Country.getElementsByTagName("State")
    #information.writerow(['PM2.5,PM10,NO2,NH3,SO2,CO,OZONE,Air Quality Index'])
    for state in State:
        #print("State:",state.getAttribute("id"))
        City = state.getElementsByTagName("City")
        for city in City:
            #print("\tCity:",city.getAttribute("id"))
            Station = city.getElementsByTagName("Station")[0]
            #print("\t\tStation:",Station.getAttribute("id"))
            #print("\t\tLast Updated:",Station.getAttribute("lastupdate"))
            Pollutant_Index = Station.getElementsByTagName("Pollutant_Index")
            for pollutants in Pollutant_Index:
                #print("\t\t\tId:", pollutants.getAttribute("id"),",Minimum Value:",pollutants.getAttribute("Min"),",Maximum value:",pollutants.getAttribute("Max"),",Average Value:", pollutants.getAttribute("Avg"))
                if(pollutants.getAttribute("Avg"))!='NA':
                    details[pollutants.getAttribute("id")] = float(pollutants.getAttribute("Avg"))
                else:
                    details[pollutants.getAttribute("id")] = float(0)
            try:
                Air_Quality_Index = Station.getElementsByTagName("Air_Quality_Index")[0]
                details['Air Quality Index'] = float(Air_Quality_Index.getAttribute("Value"))
                #print("\t\t\tAir Quality Index:",Air_Quality_Index.getAttribute("Value"))
            except IndexError:
                details['Air Quality Index'] = float(0)
                #print("\t\t\tAir Quality Index:", None)
            for key in details:
                if(details[key]==0.0):
                    missing_field_flag = True
            if(not(missing_field_flag)):
                for key in details:
                    data_list.append(details[key])
                data_tuple_list.append(tuple(data_list))
            data_list = []
            missing_field_flag = False
    df = pd.DataFrame(data_tuple_list,columns = ['PM2.5', 'PM10', 'NO2','NH3','SO2','CO','OZONE','AirQualityIndex'])
    #print(df)
    if(count==0):
        df.to_csv(r'E:\Users\user\Desktop\Air Quality123.csv')
    else:
            df.to_csv(r'E:\Users\user\Desktop\Air Quality123.csv',header = False,mode='a')

Make_Dataframe(r'E:\Users\user\Desktop\Air_quality_xml_files\Air_quality.xml')
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality1.xml",1)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality2.xml",2)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality3.xml",3)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality4.xml",4)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality5.xml",5)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality6.xml",6)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality7.xml",7)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality8.xml",8)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality9.xml",9)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality10.xml",10)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality11.xml",11)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality12.xml",12)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality13.xml",13)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality14.xml",14)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality15.xml",15)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality16.xml",16)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality17.xml",17)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality18.xml",18)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality19.xml",19)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality20.xml",20)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality21.xml",21)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality22.xml",22)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality23.xml",23)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality24.xml",24)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality25.xml",25)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality26.xml",26)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality27.xml",27)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality28.xml",28)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality29.xml",29)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality30.xml",30)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality31.xml",31)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality32.xml",32)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality33.xml",33)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality34.xml",34)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality35.xml",35)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality36.xml",36)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality37.xml",37)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality38.xml",38)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality39.xml",39)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality40.xml",40)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality41.xml",41)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality42.xml",42)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality43.xml",43)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality44.xml",44)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality45.xml",45)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality46.xml",46)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality47.xml",47)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality48.xml",48)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality49.xml",49)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality50.xml",50)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality51.xml",51)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality52.xml",52)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality53.xml",53)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality54.xml",54)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality55.xml",55)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality56.xml",56)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality57.xml",57)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality58.xml",58)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality59.xml",59)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality60.xml",60)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality61.xml",61)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality62.xml",62)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality63.xml",63)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality64.xml",64)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality65.xml",65)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality66.xml",66)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality67.xml",67)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality68.xml",68)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality69.xml",69)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality70.xml",70)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality71.xml",71)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality72.xml",72)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality73.xml",73)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality74.xml",74)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality75.xml",75)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality76.xml",76)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality77.xml",77)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality78.xml",78)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality79.xml",79)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality80.xml",80)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality81.xml",81)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality82.xml",82)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality83.xml",83)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality84.xml",84)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality85.xml",85)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality86.xml",86)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality87.xml",87)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality88.xml",88)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality89.xml",89)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality90.xml",90)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality91.xml",91)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality92.xml",92)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality93.xml",93)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality94.xml",94)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality95.xml",95)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality96.xml",96)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality97.xml",97)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality98.xml",98)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality99.xml",99)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality100.xml",100)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality101.xml",101)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality102.xml",102)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality103.xml",103)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality104.xml",104)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality105.xml",105)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality106.xml",106)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality107.xml",107)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality108.xml",108)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality109.xml",109)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality110.xml",110)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality111.xml",111)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality112.xml",112)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality113.xml",113)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality114.xml",114)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality115.xml",115)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality116.xml",116)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality117.xml",117)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality118.xml",118)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality119.xml",119)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality120.xml",120)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality121.xml",121)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality122.xml",122)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality123.xml",123)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality124.xml",124)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality125.xml",125)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality126.xml",126)
Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality127.xml",127)
#Make_Dataframe(r"E:\Users\user\Desktop\Air_quality_xml_files\Air_quality128.xml",128)
'''
dataset = np.loadtxt('Air Quality123.csv',delimiter=",",dtype='str')
print(dataset)
count_list_no = 0
try:
    for data_list in dataset:
        if(count_list_no)==0:
            count_list_no += 1
            continue
        else:
            print(count_list_no)
            for element in data_list:
                print(float(element))
except ValueError:
    print("Value Error")
'''
        
        

