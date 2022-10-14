import streamlit as st
import helper_bert



st.header('Setence acceptability (grammaticality) ')

q1 = st.text_input('Enter any Sentence')


if st.button('Check'):
    result = helper_bert.check_similarity(q1)
    #result = model.predict(query)[0]

    if result:

        st.header('acceptable')
    else:
        st.header('unacceptable')


