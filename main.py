import openai
from PIL import Image
import streamlit as st
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
import numpy as np
from streamlit_cropper import st_cropper
import urllib.request
openai.api_key = st.secrets["API_SECRET"] 

st.set_page_config( page_title="ChatGPT + DALL-E 2", page_icon="✨", layout="wide", initial_sidebar_state="auto", )

@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def openai_completion(prompt):
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt=prompt,
      max_tokens=150,
      temperature=0.5
    )
    return response['choices'][0]['text']

@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def openai_image(prompt1):
    response = openai.Image.create(
      prompt=prompt1,
      n=1,
      size="512x512"
    )
    image_url = response['data'][0]['url']
    return image_url

@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def openai_variation(byte_array):
    # This is the BytesIO object that contains your image data
    # byte_stream : BytesIO = image
    # byte_array = byte_stream.getvalue()
    response = openai.Image.create_variation(
    image=byte_array,
    n=3,
    size="512x512"
    )
    # st.write(response)
    image_url = response['data']
    return image_url

@st.cache(persist=True,allow_output_mutation=True,show_spinner=False,suppress_st_warning=True)
def openai_modify(byte_array_img,byte_array_mask,modify_text):
    response = openai.Image.create_edit(
        image=byte_array_img,
        mask= byte_array_mask,
        prompt=modify_text,
        n=1,
        size="512x512"
    )
    image_url = response['data'][0]['url']
    return image_url

st.title("📄 ChatGPT + DALL-E 🏜 Streamlit")
format_type = st.selectbox('Choose your OpenAI magician 😉',["ChatGPT","DALL-E 2"])

# st.write(st.session_state)

if 'img_url_1' not in st.session_state:
    st.session_state['img_url_1'] = []
if 'img_url_1_var_1' not in st.session_state:
    st.session_state['img_url_1_var_1'] = []
if 'img_url_1_var_2' not in st.session_state:
    st.session_state['img_url_1_var_2'] = []
if 'img_url_1_var_3' not in st.session_state:
    st.session_state['img_url_1_var_3'] = []
if 'img_url_1_modify_1' not in st.session_state:
    st.session_state['img_url_1_modify_1'] = []

if 'img_upload_1_modify_1' not in st.session_state:
    st.session_state['img_upload_1_modify_1'] = []
if 'img_upload_1_var_1' not in st.session_state:
    st.session_state['img_upload_1_var_1'] = []
if 'img_upload_1_var_2' not in st.session_state:
    st.session_state['img_upload_1_var_2'] = []
if 'img_upload_1_var_3' not in st.session_state:
    st.session_state['img_upload_1_var_3'] = []

if format_type == "ChatGPT":
    input_text = st.text_area("Please enter text here... 🙋",height=50)
    chat_button = st.button("Do the Magic! ✨")

    if chat_button and input_text.strip() != "":
        with st.spinner("Loading...💫"):
            openai_answer = openai_completion(input_text)
            st.success(openai_answer)
    else:
        st.warning("Please enter something! ⚠")

else:
    
    upload_img = st.checkbox("upload reference image")
    if upload_img:
        bg_image = st.file_uploader("Background image:", type=["png", "jpg"])
        
        if bg_image:
            
            original_upload_img = bg_image
            st.set_option('deprecation.showfileUploaderEncoding', False)

            Crop_image =st.checkbox("Crop Image")
            if Crop_image:
                col3,col1,col2 = st.columns([1, 2,2])
                # Upload an image and set some options for demo purposes
                st.header("Cropper Demo")
                img_file = bg_image
                box_color = '#000000'
                aspect_choice = col3.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
                aspect_dict = { "1:1": (1, 1), "16:9": (16, 9), "4:3": (4, 3), "2:3": (2, 3), "Free": None }
                aspect_ratio = aspect_dict[aspect_choice]
                if img_file:
                    img = Image.open(img_file)
                    # Get a cropped image from the frontend
                    with col1:
                        cropped_img = st_cropper(img, realtime_update=True, box_color=box_color, aspect_ratio=aspect_ratio)
                    with col2:
                        # Manipulate cropped image at will
                        st.write("Preview")
                        _ = cropped_img.thumbnail((150,150))
                        st.image(cropped_img)
                    bg_image = cropped_img 

            mode_choice2 = st.radio(label="Select Mode", options=["None","Mask", "Variation"])
            
            # upload img MASK ----------------------------
            if mode_choice2 == "Mask":
                st.info("Mark the element in the image with black color where you want to modify ")
                drawing_mode = "freedraw"
                stroke_width = st.slider("Stroke width: ", 1, 30, 20)
                stroke_color = '#000000'
                bg_color =  "#eee"
                # Resize the background image
                if Crop_image:
                    bg_image_ = bg_image
                else:
                    bg_image_ = Image.open(bg_image)
                width = 512
                bg_image_resize = bg_image_.resize(size= (width,width))
                new_image = bg_image_resize
                # Create a canvas component
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image= new_image if new_image else None,
                    update_streamlit=True,
                    height=width,
                    width = width,
                    drawing_mode=drawing_mode,
                    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
                    display_toolbar=True,
                    key="full_app",
                )

                # Do something interesting with the image data and paths
                if canvas_result.image_data is not None:
                    mask_img = canvas_result.image_data
                    # st.write("out")
                    # st.write(canvas_result.image_data.shape)
                    # st.image(canvas_result.image_data)
                    alpha = mask_img[:,:,3]
                    # st.write(alpha)

                    # Convert the image to a BytesIO object
                    bg_image_image = new_image
                    bg_image_image = bg_image_image.convert('RGBA')
                    byte_stream = BytesIO()
                    bg_image_image.save(byte_stream, format='PNG')
                    byte_array_img = byte_stream.getvalue()

                    # Convert the mask to a BytesIO object
                    bg_image_image = new_image
                    bg_image_image = bg_image_image.convert('RGBA')
                    bg_image_array = np.array(bg_image_image)
                    # st.write(bg_image_array.shape)
                    for i in range(width):
                        for j in range(width):
                            if mask_img[:,:,3][i][j]>0:
                                bg_image_array[:,:,3][i][j] = 0
                    # st.write(bg_image_array[:,:,3])
                    new_bg_image = Image.fromarray(bg_image_array)
                    # st.write(new_bg_image)
                    byte_stream_ = BytesIO()
                    new_bg_image.save(byte_stream_, format='PNG')
                    byte_array_mask = byte_stream_.getvalue()

                modify_text1 = st.text_area("Please enter text here..... 🙋",height=50)
                modify2 = st.button("Modify")
                if modify2 and modify_text1.strip() != "":
                    image_url = openai_modify(byte_array_img,byte_array_mask,modify_text1)
                    st.session_state['img_upload_1_modify_1']  = image_url
                else:
                    st.warning("Please enter something!!! ⚠")

                if st.session_state['img_upload_1_modify_1'] :
                    st.image(st.session_state['img_upload_1_modify_1'] , caption='Generated by OpenAI')

            # upload img variation  --------------------
            if mode_choice2 == "Variation":
                """ Variation """
                variation_img = st.button("Get variation of image")
                if variation_img:
                    if Crop_image:
                        image = bg_image
                    else:
                        image = Image.open(bg_image)
                    # image = Image.open(bg_image)
                    width, height = 512, 512
                    image_r = image.resize((width, height))
                    # Convert the image to a BytesIO object
                    byte_stream = BytesIO()
                    image_r.save(byte_stream, format='PNG')
                    byte_array = byte_stream.getvalue()
                    variation_url = openai_variation(byte_array)
                    st.session_state['img_upload_1_var_1'] = variation_url[0]['url']
                    st.session_state['img_upload_1_var_2'] = variation_url[1]['url']
                    st.session_state['img_upload_1_var_3'] = variation_url[2]['url']

                if st.session_state['img_upload_1_var_3']:
                    col1,col2,col3 = st.columns(3)
                    col1.image(st.session_state['img_upload_1_var_1'], caption='Variation 1') 
                    col2.image(st.session_state['img_upload_1_var_2'], caption='Variation 2') 
                    col3.image(st.session_state['img_upload_1_var_3'], caption='Variation 3') 
                    
                    image_choice1 = st.radio(label="Select Image", options=["Variation 1", "Variation 2", "Variation 3"])
                    aspect_dict = { "Variation 1" : st.session_state['img_upload_1_var_1'] ,
                                    "Variation 2" : st.session_state['img_upload_1_var_2'], 
                                    "Variation 3": st.session_state['img_upload_1_var_3'] 
                                }
                    selected_img_url1 = aspect_dict[image_choice1]



    # generate from text --------------------------------------------------------------------------------------------------
    else:
        input_text = st.text_area("Please enter text here... 🙋",height=50)
        image_button = st.button("Generate Image 🚀")
        if image_button and input_text.strip() != "":
            image_url_1 = openai_image(input_text)
            st.session_state['img_url_1'] = image_url_1
        else:
            st.warning("Please enter something! ⚠")
            
        if st.session_state['img_url_1']:
            st.image(st.session_state['img_url_1'], caption='Generated by OpenAI') 

        if  st.session_state['img_url_1']:
            urllib.request.urlretrieve(st.session_state['img_url_1'], "image_url_1.png")
            img_url_1_open = Image.open("image_url_1.png")
            
            mode_choice1 = st.radio(label="Select Mode", options=["None","Mask", "Variation"])

            if mode_choice1 == "Mask":
                """ MASK"""
                st.info("Mark the element in the image with black color where you want to modify ")
                drawing_mode = "freedraw"
                stroke_width = st.slider("Stroke width: ", 1, 30, 20)
                stroke_color = '#000000'
                bg_color =  "#eee"
                # Resize the background image
                bg_image_ = img_url_1_open
                width = 512
                bg_image_resize = bg_image_.resize(size= (width,width))
                new_image = bg_image_resize
                # Create a canvas component
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                    stroke_width=stroke_width,
                    stroke_color=stroke_color,
                    background_color=bg_color,
                    background_image= new_image if new_image else None,
                    update_streamlit=True,
                    height=width,
                    width = width,
                    drawing_mode=drawing_mode,
                    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
                    display_toolbar=True,
                    key="full_app",
                )

                # Do something interesting with the image data and paths
                if canvas_result.image_data is not None:
                    mask_img = canvas_result.image_data
                    # st.write("out")
                    # st.write(canvas_result.image_data.shape)
                    # st.image(canvas_result.image_data)
                    alpha = mask_img[:,:,3]
                    # st.write(alpha)

                    # Convert the image to a BytesIO object
                    bg_image_image = new_image
                    bg_image_image = bg_image_image.convert('RGBA')
                    byte_stream = BytesIO()
                    bg_image_image.save(byte_stream, format='PNG')
                    byte_array_img = byte_stream.getvalue()

                    # Convert the mask to a BytesIO object
                    bg_image_image = new_image
                    bg_image_image = bg_image_image.convert('RGBA')
                    bg_image_array = np.array(bg_image_image)
                    # st.write(bg_image_array.shape)
                    for i in range(width):
                        for j in range(width):
                            if mask_img[:,:,3][i][j]>0:
                                bg_image_array[:,:,3][i][j] = 0
                    # st.write(bg_image_array[:,:,3])
                    new_bg_image = Image.fromarray(bg_image_array)
                    # st.write(new_bg_image)
                    byte_stream_ = BytesIO()
                    new_bg_image.save(byte_stream_, format='PNG')
                    byte_array_mask = byte_stream_.getvalue()

                modify_text = st.text_area("Please enter text here.... 🙋",height=50)
                modify1 = st.button("Modify")
                if modify1 and modify_text.strip() != "":
                    image_url = openai_modify(byte_array_img,byte_array_mask,modify_text1)
                    st.session_state['img_url_1_modify_1']  = image_url
                else:
                    st.warning("Please enter something!! ⚠")

                if st.session_state['img_url_1_modify_1'] :
                    st.image(st.session_state['img_url_1_modify_1'] , caption='Generated by OpenAI')

            if mode_choice1 == "Variation":
                """ Variation """
                variation_img = st.button("Get variation of image")
                if variation_img:
                    image = img_url_1_open
                    width, height = 512, 512
                    image_r = image.resize((width, height))
                    # Convert the image to a BytesIO object
                    byte_stream = BytesIO()
                    image_r.save(byte_stream, format='PNG')
                    byte_array = byte_stream.getvalue()
                    variation_url = openai_variation(byte_array)
                    st.session_state['img_url_1_var_1'] = variation_url[0]['url']
                    st.session_state['img_url_1_var_2'] = variation_url[1]['url']
                    st.session_state['img_url_1_var_3'] = variation_url[2]['url']

                if st.session_state['img_url_1_var_3']:
                    col1,col2,col3 = st.columns(3)
                    col1.image(st.session_state['img_url_1_var_1'], caption='Variation 1') 
                    col2.image(st.session_state['img_url_1_var_2'], caption='Variation 2') 
                    col3.image(st.session_state['img_url_1_var_3'], caption='Variation 3') 
                image_choice1 = st.radio(label="Select Image", options=["Variation 1", "Variation 2", "Variation 3"])
                aspect_dict = { "Variation 1" : st.session_state['img_url_1_var_1'] ,
                                "Variation 2" : st.session_state['img_url_1_var_2'], 
                                "Variation 3": st.session_state['img_url_1_var_3'] 
                            }
                selected_img_url1 = aspect_dict[image_choice1]

    # if not variation_img:
    #     if image_button and input_text.strip() != "":
    #         with st.spinner("Loading...💫"):
    #             # Convert the image to a BytesIO object

    #             # mask_img_image = Image.fromarray(np.uint8(cm.gist_earth(mask_img)*255))
    #             # out = Image.convert("RGBA", mask_img)  
    #             # st.write(out)
    #             # byte_stream_ = BytesIO()
    #             # im = Image.fromarray(mask_img, mode="RGBA")
    #             # st.write(im)
    #             # im.save(byte_stream_, format='PNG')
    #             # byte_array_mask = byte_stream_.getvalue()
    #             if mask_image:
    #                 response = openai.Image.create_edit(
    #                 image=byte_array_img,
    #                 mask= byte_array_mask,
    #                 prompt=input_text,
    #                 n=1,
    #                 size="512x512"
    #                 )
    #                 image_url = response['data'][0]['url']
    #                 st.image(image_url, caption='Generated by OpenAI')
  

