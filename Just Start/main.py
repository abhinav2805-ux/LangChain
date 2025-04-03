import streamlit as st
import langchain_helper


st.title("Restaurant Name Generator")

cuisine = st.sidebar.selectbox("Pick a Cuisine" , ("Indian" , "Mexican" , "Italian" , "Chinese" , "American"))





if cuisine:
    response = langchain_helper.generate_restaurant_and_menu(cuisine)
    
    st.header(response['restaurant_name'].strip())
    menu_items = response['menu'].strip().split(',')
    st.write("----------- Menu Items ----------------")
    
    for item in menu_items:
        st.write("-",item)