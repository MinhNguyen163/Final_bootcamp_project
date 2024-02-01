import streamlit as st
import pandas as pd
import numpy as np 
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Unlocking Insights:')
st.header('House price prediction in Netherlands', divider='rainbow')

st.text('This is a web app demo to check house price in NL')

#st.markdown('## This is **markdown**')

#uploaded_file = st.file_uploader('Upload your file here')
house_final = pd.read_csv('../Final_project_data/5_house_final_all.csv')
house_final['house_id'] = house_final['house_id'].astype(str)

#st.write(house_final.head(2))

city = str(st.selectbox('Which city that you would like to check: ', ['Amsterdam', 'Almere', 'Eindhoven', 'Leeuwarden', 'Rotterdam', 'Utrecht', 'Nijmegen', 'Masstricht'])).lower()
user_match = house_final[house_final['city'] == city]
#st.write(city)
#st.write(user_match.shape)

house_type = str(st.selectbox('Any preference on house types that you would like to check: ', ['no preference', 'Apartement', 'Huis'])).lower()
if house_type != 'no preference':
    user_match = user_match[user_match['house_type'] == house_type]
else:
    user_match = user_match
#st.write(user_match.shape)
building_type = str(st.selectbox('Any preference on building types that you would like to check: ', ['no preference', 'Bestaande bouw', 'Nieuwbouw']))
if building_type != 'no preference':
    user_match = user_match[user_match['building_type'] == building_type]
else:
    pass
#st.write(user_match.shape)
room = int(st.select_slider('Total number of rooms: ', [1,2,3,4,5,6,7,8]))
user_match = user_match[user_match['room'] == room ]
#st.write(user_match.shape)

left_col, right_col = st.columns(2)
living_area_min = float(left_col.select_slider('Minimum living area is: ', range(10,511)))
living_area_max = float(right_col.select_slider('Maximum living area is: ', range(10,511)))
user_match = user_match[(user_match['living_area']<=living_area_max) & (user_match['living_area']>=living_area_min) ]
#st.write(user_match.shape)
energy_label = st.selectbox('Any preferences on energy label of the house: ', ['no preference', '>A+', 'A', 'B', 'C', 'D', 'E', 'G'])
if energy_label != 'no preference':
    user_match = user_match[user_match['energy_label'] == energy_label]
else:
    pass

user_match_1 = user_match[['city', 'house_type', 'building_type', 'room', 'bedroom', 'bathroom', 'living_area', 'energy_label', 'zip', 'year_built', 'city_price_ratio']]
#st.write(user_match.head())
#st.write(user_match_1.shape)

#control the number of return records that match user criteria

#try:
# load label encoder
path = '../encoders/'
filename = 'label_encoder_for_zip.pkl'
with open(path+filename, 'rb') as file:
    le = pickle.load(file)
#st.write([le.classes_])

# convert zip from label encoder 
user_match_1['zip'] = le.transform(user_match_1['zip'])
#st.write(user_match_1['zip'])

# split num vs cat
user_match_1_cat = user_match_1.select_dtypes(object)
user_match_1_num = user_match_1.select_dtypes(np.number)

# load onehot encoder
path = '../encoders/'
filename = 'one_hot_encoder.pkl'
with open(path+filename, 'rb') as file:
    encoder = pickle.load(file)

# encode cat variables
user_match_1_cat_encoded = encoder.transform(user_match_1_cat).toarray()
user_match_1_cat_encoded_df = pd.DataFrame(user_match_1_cat_encoded, columns = encoder.get_feature_names_out(), index = user_match_1_cat.index)

# load power transformer and apply
path = '../transformer/'
filename = 'power_transformer.pkl'
with open(path+filename, 'rb') as file:
    pt = pickle.load(file)

# transform num variables
user_match_1_num_pt = pt.transform(user_match_1_num)
user_match_1_num_pt_df = pd.DataFrame(user_match_1_num_pt, columns = user_match_1_num.columns, index = user_match_1_num.index)

# combine cat and num df after encode and transform
user_match_1 = pd.concat([user_match_1_cat_encoded_df, user_match_1_num_pt_df], axis = 1)

# load minmaxscale and apply
#path = '../scaler/'
#filename = 'min_max_scaler.pkl'
#with open(path+filename, 'rb') as file:
#    scaler = pickle.load(file)

# apply minmax scale to the whole df
#user_match_1_scaled = scaler.transform(user_match_1)
#user_match_1_scaled_df = pd.DataFrame(user_match_1_scaled, columns = user_match_1.columns, index = user_match_1.index)
#st.write(user_match_1_scaled_df.head())

# load XGB_best model and apply for price predictions 
path = '../models/'
filename = 'XGB_best_all_final.pkl'
with open(path+filename, 'rb') as file:
    XGB_best = pickle.load(file)
#except:
    #st.warning('There is no matching house in our current record. Please adjust the criteria')

price_pred = XGB_best.predict(user_match_1).round(0)
#st.write(price_pred)
#st.write(user_match_1.index)
result = user_match.loc[user_match_1.index, :]
#st.write(result['zip'])
#result['zip']=le.inverse_transform(result['zip']).astype(str)
result['price_pred'] = price_pred = XGB_best.predict(user_match_1).round(0)
#st.write(result['city'].unique())

st.subheader('Below is the _:green[predicted price range]_ shown as a boxplot across the zipcode in your selected city', divider='rainbow')
plt.figure(figsize=(12, 8))
sns.boxplot(x='zip', y='price_pred', data=result[['zip','price_pred']])
plt.ylim([200000, 1000000])
plt.title('House Prices by Zip Code')
plt.xlabel('Zip Code')
plt.ylabel('Price')
plt.axhline(y=400000, linestyle='--', color='r')
plt.axhline(y=500000, linestyle='--', color='g')
plt.axhline(y=600000, linestyle='--', color='b')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.show()

st.pyplot(plt.gcf())

result['difference_pred_real'] = result['price_pred'] - result['price']
result['ratio_difference_pred_real'] = (result['price_pred']/result['price']).round(2)
result_display = result[result['ratio_difference_pred_real']>=1.10]
result_display = result_display[['house_id', 'city','price', 'price_pred', 'difference_pred_real', 'ratio_difference_pred_real', 'house_type', 'building_type', 'room', 'bedroom', 'bathroom', 'living_area', 'energy_label', 'zip', 'year_built']]
result_display.sort_values(by = 'ratio_difference_pred_real',inplace = True, ascending=False)
#st.subheader('This is a subheader with a divider', divider='rainbow')
#st.subheader('_Streamlit_ is :blue[cool] :sunglasses:')
st.subheader('For :red[investment], please have a look at our :green[suggestion] below!', divider='rainbow')
st.subheader('Here is the list of _:blue[top 10 houses]_ that have predicted price higher than the listed price \n(with minimum 10\% difference)')

st.write(result_display.head(10))
#st.data_editor(result, column_config={'photo':st.column_config.LinkColumn('photo' )})
# display map or plot for price predictions range base on zip
# and maybe? suggest for house with a predicted price higher than the listed price --> value gain? for investment?


