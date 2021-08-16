import streamlit as st
import pandas as pd
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import plotly.express as px


def predict(image1,meta_data):
    IMAGE_SHAPE = (380, 380, 3)
    model = load_model("cls_weights.hdf5", compile=False)
    test_image = image1.resize((380, 380))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['Squamous Cell Carcinoma','Pigmented Benign Keratosis' , 'Actinic Keratosis']
    predictions = model.predict([meta_data, test_image]).reshape(3)
    predictions = (predictions*100).round(2)
    title = class_names[np.argmax(predictions)]
    fig = px.bar(x= class_names,y= predictions)
    fig.update_layout(
        title="Classification Result",
        xaxis_title="Lesions",
        yaxis_title="Probability",
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="RebeccaPurple")
    )

    return fig


st.markdown(
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" '
    'integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
    unsafe_allow_html=True,
)
query_params = st.experimental_get_query_params()
tabs = ["Home", "EDA", "Model", "Resources and Article"]
if "tab" in query_params:
    active_tab = query_params["tab"][0]
else:
    active_tab = "Home"

if active_tab not in tabs:
    st.experimental_set_query_params(tab="Home")
    active_tab = "Home"

li_items = "".join(
    f"""
    <li class="nav-item">
        <a class="nav-link{' active' if t == active_tab else ''}" href="/?tab={t}">{t}</a>
    </li>
    """
    for t in tabs
)
tabs_html = f"""
    <ul class="nav nav-tabs">
    {li_items}
    </ul>
"""

st.markdown(tabs_html, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if active_tab == "Home":
    st.write("## Welcome to Skin Disorders Detection Application for Three Different Lesions")
    im1 = Image.open(r"Data/combined_les.jpg")
    st.image(im1)
    st.markdown("(This project has made during the Master1 Internship in Challenge Hub at CRI.) ")
    st.markdown("### ❓ General Problem:")
    st.markdown(
        "According to World Health Organization, skin diseases are among the most common of all human health "
        "afflictions and affect almost 900 million people in the world at any time.  Acne, mycosis, herpes, "
        "atopic dermatitis, eczema, so many various forms which can have consequences on the quality of life of the "
        "people who suffer from them. As reported by Derma Survey in 2013, the average waiting time for regular "
        "visits is 36 days in Europe. If we look for France, it is over 20 days. Detection and follow-up of "
        "dermatological diseases takes long days and patients wear out as psychology in this process.")
    im2 = Image.open(r"Data/derma.jpg")
    st.image(im2)
    st.markdown("### 🎯 Goals")
    st.markdown(
        "The general goal of this project is to simulate how we can detect skin diseases with computer vision "
        "technology by starting 3 different lesions. And to achieve the best accuracy by comparing existing "
        "artificial neural network algorithm models.")
    st.markdown("It is aimed to be done in the following order.")
    st.markdown("* 1. Improving Computer Vision Skills")
    st.markdown("* 2. Detecting of Three Lesions with Good Accuracy")
    st.markdown("* 3. Creating an interface for easily testing")
    st.markdown("### ⏭️Future Goal")
    st.markdown("To provide that artificial intelligence takes place in a large part of the detection and treatment "
                "processes by accelerating data collection studies. And thereby alleviating the workload of "
                "dermatologists. To be able to detect all skin diseases with computer systems instead of detecting "
                "them visually.")
    st.markdown("The biggest problem at this point is data collection. How can we do this? First of all, "
                "various diseases need to be labeled by dermatologists. Therefore, by producing a device or "
                "application that will facilitate their work, they enable them to follow their patients and to "
                "remember the cases better, sample data are formed in their hands. Later, when the accuracy rates "
                "begin to increase, it is opened to the use of general doctors with an application and the first "
                "detection and treatment application is progressed by the general doctors for a while. When our "
                "database reaches saturation and a sufficient degree of accuracy and the public's confidence "
                "increases, the diagnosis and treatment process at home for simple diseases can be started directly "
                "in the public application.")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("** Resources ** : ")
    st.markdown("(1) https://www.who.int/neglected_diseases/news/WHO-publishes-pictorial-training-guide-on-neglected"
                "-skin-disease/en/ ")
    st.markdown("(2) https://www.dermasurvey.eu/derma-survey/ ")




elif active_tab == "Model":
    st.sidebar.title("Selector")
    st.sidebar.write("-----------")
    st.sidebar.write("**Usage of Sidebar** ")
    st.sidebar.write("First you need to fill age,clinical anatom site general and sex meta data and then upload image "
                     "finally you can classify by clicking button at the bottom")

    age_select = st.sidebar.selectbox(" What is your age range?",("None","0-16","16-32","32-48","48-64","48-64",">64"))
    st.sidebar.write('<span style="color:%s">%s</span>' % ('red', " You selected: "+age_select), unsafe_allow_html=True)
    sex_select = st.sidebar.selectbox(" What is your age gender?", ("None","Female", "Male"))
    st.sidebar.write('<span style="color:%s">%s</span>' % ('red', " You selected: "+sex_select), unsafe_allow_html=True)
    anatom_select = st.sidebar.selectbox(" Where is the lesion on the body?", ("None","Anterior torso", "Lower extremity", "Upper extremity", "Head/neck", "Palms/soles", "Posterior torso"))
    st.sidebar.write('<span style="color:%s">%s</span>' % ('red', " You selected: " + anatom_select),unsafe_allow_html=True)
    file_uploaded = st.sidebar.file_uploader("Choose File", type=["png", "jpg", "jpeg"])
    class_btn = st.sidebar.button("Classify Skin Disorder")



    st.markdown("In this section we will classify our skin disorders by just uploading image. ")
    st.markdown("After uploading image it will give a chart that include three bar to show classifying result with "
                "percentage.")
    st.markdown("Please upload your image from sidebar, which can only have three different lesions: * Pigmented Benign Keratosis, "
                "Squamous Cell Carcinoma, Actinic Keratosis * .")
    st.markdown("(If you want to see model info and evaluation you can just click below button.)")



    col1, col2, col3 = st.beta_columns([1,1,1])
    age = 0
    anatom = 0
    sex = 0
    if anatom_select=="Anterior torso":
        im_anterior = Image.open(r"Data\Anterior torso.jpg")
        r_im_anterior = im_anterior.resize((205, 480))
        col2.image(r_im_anterior)
        anatom = 1.0
    elif anatom_select=="Lower extremity":
        im_lower = Image.open(r"Data\Lower extremity.jpg")
        r_im_lower = im_lower.resize((205, 480))
        col2.image(r_im_lower)
        anatom = 2.0
    elif anatom_select=="Upper extremity":
        im_upper = Image.open(r"Data\Upper extremity.jpg")
        r_im_upper = im_upper.resize((205, 480))
        col2.image(r_im_upper)
        anatom = 3.0
    elif anatom_select=="Head/neck":
        im_h = Image.open(r"Data\Head_neck.jpg")
        r_im_h = im_h.resize((205, 480))
        col2.image(r_im_h)
        anatom = 0.0
    elif anatom_select=="Palms/soles":
        im_p = Image.open(r"Data\_Palms_soles.jpg")
        r_im_p = im_p.resize((205, 480))
        col2.image(r_im_p)
        anatom = 5.0
    elif anatom_select=="Posterior torso":
        im_pos = Image.open(r"Data\Posterior torso.jpg")
        r_im_pos = im_pos.resize((205, 480))
        col2.image(r_im_pos)
        anatom = 4.0

    if age_select == "0-16":
        age  = 0.0
    elif age_select == "16-32":
        age = 1.0
    elif age_select == "32-48":
        age = 2.0
    elif age_select == "45-64":
        age = 3.0
    elif age_select == ">64":
        age = 4.0

    if sex_select == "Female":
        sex = 1.0
    elif sex_select == "Male":
        sex = 0.0
    if file_uploaded is not None:
        image_upload = Image.open(file_uploaded)
        st.image(image_upload, caption='Uploaded Image', use_column_width=True)
    d_f = {'meta.clinical.age_approx':[age],'meta.clinical.anatom_site_general':[anatom],'meta.clinical.sex':[sex]}
    d_f = pd.DataFrame.from_dict(d_f)
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                #gray_img = cv2.cvtColor(image_upload, cv2.COLOR_BGR2GRAY)
                #heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_HSV)

                plt.imshow(image_upload)
                plt.axis("off")
                predictions = predict(image_upload,d_f)
                time.sleep(1)
                st.success('Classified')
                st.plotly_chart(predictions)
                st.write("**Ground Truth**" )
                st.success('Correct! Pigmented Benign Keratosis')
                #plt.imshow(heatmap_img)
                heat = Image.open(r"Data/image_heat.png")
                co1, co2 = st.beta_columns([1,1])
                co1.image(image_upload, use_column_width=True)
                co2.image(heat, use_column_width=True)
    if  anatom_select == 'None' and sex_select=='None' and age_select=="None":
        st.error(" ⚠️ Please Use Sidebar")

    st.markdown("")
    st.markdown("")
    st.markdown("Click to see evaluation...")
    with st.beta_expander("Model Evaluation"):
        ev = Image.open(r"Data/confusion.png")
        ev1 =Image.open(r"Data/precision.png")
        st.image(ev1, use_column_width=True)
        st.image(ev, use_column_width=True)






elif active_tab == "Resources and Article":
    st.markdown("## Three different skin lesions classification using multiply input on combined Net, EfficientNet "
                "and MLP.")
    st.markdown("*Yunus Emre Celik -CRI University of Paris Digital Science Master*")
    st.write("----------")
    st.markdown("### Introduction")
    st.markdown("Skin disorders are a type of disease that starts from benign scars and extends to the disease that "
                "causes permanent damage and even to skin cancer. Rapid and automated skin lesion detection holds "
                "great promise for early diagnosis and prevention with the development of computer vision systems. In "
                "this project, starting with simple artificial neural network models that can detect these diseases, "
                "continuing with complex transfer learning models which are EfficientNet,Resnet, and finally "
                "combining the two inputs with the combined method,it has been ensured that 3 diseases are output. ")
    st.markdown("")
    st.markdown("")
elif active_tab == "EDA":
    st.markdown("## Exploratory Data Analysis ")
    st.markdown("(Images and Metadata provided by ISIC-archive:https://www.isic-archive.com)")
    st.markdown("Let's look at together our data and images...")
    st.markdown("### Images")
    st.markdown("")

    col1, col2, col3 = st.beta_columns(3)

    im2 = Image.open(r"Data\Picture1.png")
    im2 = im2.resize((112, 112))
    col1.markdown("**Pigmented Benign Keratosis(pbk)**")
    col1.image(im2, use_column_width=True)
    col1.markdown(" Pigmented skin lesions are extremely common, with almost all patients having a number of pigmented "
                  "lesions on their skin. When considering their various characteristics, it is useful to divide "
                  "these lesions into melanocytic, keratinocytic, vascular and reactive lesions.")
    im3 = Image.open(r"Data\Picture2.png")
    im3 = im3.resize((110, 105))
    col2.markdown("**Squamous Cell Carcinoma(scc)**")
    col2.image(im3, use_column_width=True)
    col2.markdown("Squamous cell carcinoma of the skin is a common form of skin cancer that develops in the squamous "
                  "cells that make up the middle and outer layers of the skin.")
    im4 = Image.open(r"Data\Picture3.png")
    im4 = im4.resize((111, 111))
    col3.markdown("**Actinic Keratosis(ak)**")
    col3.markdown("")
    col3.image(im4, use_column_width=True)
    col3.markdown("Actinic keratoses (also called solar keratoses) are dry scaly patches of skin that have been "
                  "damaged by the sun. The patches are not usually serious. But there's a small chance they could "
                  "become skin cancer, so it's important to avoid further damage to your skin.")
    st.markdown("")
    st.markdown("")
    st.markdown("##### Ratio of Lesion's Amount")
    im5 = Image.open(r"Data\Picture5.png")
    st.image(im5)
    st.markdown(" We have total : **2624** images which include and split : pbk = 1099, scc = 656, ak = 869.")
    st.markdown("##### Meta Data Explore")
    im6 = Image.open(r"Data/meta_data.png")
    st.image(im6)
    st.write("-----------")
    st.markdown("As we can see this graph there is a lot of null column so we can easily drop them.")
    im7 = Image.open(r"Data/clean_column.png")
    st.image(im7)
    st.write("-----------")
    st.markdown("There are still some missing values on columns but we will fill those values after looking "
                "distributi of columns values. ")
    st.markdown("Let's look at distributions of columns.")
    im8 = Image.open(r"Data/dist1.png")
    st.image(im8)
    im9 = Image.open(r"Data/dist2.png")
    st.image(im9)
    im10 = Image.open(r"Data/dist3.png")
    st.image(im10)
    st.write("-----------")
    st.markdown(" We should fill missing value or delete before to look at Feature Important.")
    im11 = Image.open(r"Data/dist4.png")
    st.image(im11)
    st.markdown("This graph shows us where is the missing values. So we fill for age with median because there is no "
                "mising values on that. For sex column we will fill with male because of most of them male,"
                "confirm type are mostly hispotology, anatom site general are head/neck so we can fill most usued "
                "values because it will not cause bias.  ")
    im12 = Image.open(r"Data/dist5.png")
    st.image(im12)
    st.write("-----------")
    st.markdown("After fill missing values we need to change all columns to numerical values to look at correlation "
                "and feature important.")
    im13 = Image.open(r"Data/dit6.png")
    st.image(im13)
    st.write("-----------")
    st.markdown("The most correlate with diagnosis is anatom site general column . So it is important where the skin "
                "lesion on the body.")
    st.markdown("Now we will look feature important according to diagnosis with XGBClassifier. ")
    im14 = Image.open(r"Data/dist7.png")
    st.image(im14)
    st.write("-----------")
    st.markdown("According to feature importants we can use for MLP combined model just three of them these are will "
                "be Age, Sex and Anatom. Confirm Type is making with doctor or histology so we can not ask patents "
                "to take these result.")
    st.markdown("#### Let's pass model section....")






else:
    st.error("Something has gone terribly wrong.")



