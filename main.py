import pandas as pd
import streamlit as st
from bertopic_model import *
import os
import streamlit.components.v1 as components
from umap import UMAP
import dill


# Set page title
st.set_page_config(page_title='Tamil Topic Modelling Tool', layout="wide")

st.title('Tamil Topic Modelling Tool')


def delete_files():
    for f in os.listdir(file_path):
        os.remove(os.path.join(file_path, f))

    for f in os.listdir("./templates/"):
        os.remove(os.path.join("./templates/", f))


# Create file uploader for data
data_file = st.file_uploader('Upload dataset', type=[
                             'csv'], on_change=delete_files)

st.sidebar.write('**Pre topic modelling**')
# Create topic modeling options

top_n_words = st.sidebar.slider(
    'Top n words per topic', min_value=2, max_value=50, step=1, value=10)

reduce_topics = st.sidebar.checkbox('Reduce number of topics to ', value=False)
nr_topics = None
if reduce_topics:
    nr_topics = st.sidebar.slider(
        'Number of topics to be reduced to', min_value=2, max_value=20, step=1)


low_memory = st.sidebar.checkbox('Run on low memory', value=False)


# Check if data has been uploaded
if data_file is not None:

    # Read data into a DataFrame
    df = pd.read_csv(data_file)
    # Display data
    st.write(df)
    # Create topic modeling button
    if st.button('Run topic modeling'):
        with st.spinner('Running...'):

            set_topic_model(top_n_words, nr_topics, low_memory)
            run(df, top_n_words, nr_topics, low_memory)
        st.write('Topic modeling complete')

    with st.sidebar:

        st.write('**Post topic modelling**')
        pos_removal = st.sidebar.checkbox(
            'Fine-tune by removing particular POS tags', value=False)

        tags = []
        if pos_removal:
            tags = st.sidebar.multiselect('Select POS tags to remove', [
                'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'])
            if len(tags) == 0:
                st.sidebar.error('Please select at least one POS tag')
            else:
                apply_pos_removal(df, tags)
                st.write('POS tags removed')

        if st.button('Reduce outliers'):
            reduce_outliers(df)
            st.write('Outliers reduced')

        if st.button('Get topic info'):
            st.write(get_topic_info())

        search = st.text_input('Search topic')

        if search:
            st.write(search_topic(search, 5))

        def visualize():
            with st.spinner('Visualizing topics...'):

                trained_model = load()

                docs = df['text'].apply(
                    clean).tolist()
                hierarchical_topics = trained_model.hierarchical_topics(docs)

                embeddings = load_embeddings()

                fig1 = trained_model.visualize_topics()

                reduced_embeddings = UMAP(
                    n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)

                fig2 = trained_model.visualize_documents(
                    docs, reduced_embeddings=reduced_embeddings)

                fig3 = trained_model.visualize_hierarchy(
                    hierarchical_topics=hierarchical_topics)

                fig4 = trained_model.visualize_barchart()

                fig5 = trained_model.visualize_heatmap()

                fig1.write_html("./templates/fig1.html")
                fig2.write_html("./templates/fig2.html")
                fig3.write_html("./templates/fig3.html")
                fig4.write_html("./templates/fig4.html")
                fig5.write_html("./templates/fig5.html")

        st.button('Visualize topics', on_click=visualize)

    # check if files exist
    if os.path.exists("./templates/fig1.html"):
        p1 = open("./templates/fig1.html", "r")
        components.html(p1.read(),  height=700)

    if os.path.exists("./templates/fig2.html"):
        p2 = open("./templates/fig2.html", "r")
        components.html(p2.read(),  height=700)

    if os.path.exists("./templates/fig3.html"):
        p3 = open("./templates/fig3.html", "r")
        components.html(p3.read(),  height=700)

    if os.path.exists("./templates/fig4.html"):
        p4 = open("./templates/fig4.html", "r")
        components.html(p4.read(),  height=700)

    if os.path.exists("./templates/fig5.html"):
        p5 = open("./templates/fig5.html", "r")
        components.html(p5.read(),  height=800)
