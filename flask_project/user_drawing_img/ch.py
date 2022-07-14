import pandas as pd
csv_test = pd.read_csv('C:\\Users\\qorgh2akfl\\Desktop\\flask_server\\df.csv')
#print(csv_test)

str = "롯데월드 어드벤처"



#print(csv_test.columns)
# print(csv_test["name"][1])
# print(len(csv_test["name"]))
#print(len(csv_test.columns))

# for i in range(0,len(csv_test["name"])):
#     #print(len(csv_test["name"][i]))
#     label.append(csv_test["name"][i])
# print(label)
# num=label.index(str)

f_row=csv_test.loc[csv_test["name"]==str]
name=f_row["name"].astype("string")
tel = f_row["tel"].astype("string")
address = f_row["address"].astype("string")
latitude = f_row["latitude"].astype("string")
longitude =f_row["longitude"].astype("string")


print(name[1],tel[1],address[1],latitude[1],longitude[1])

