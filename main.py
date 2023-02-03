import openai
from PIL import Image
import streamlit as st
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
import numpy as np
from streamlit_cropper import st_cropper

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
def openai_image(prompt):
    response = openai.Image.create(
      prompt=prompt,
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

st.title("📄 ChatGPT + DALL-E 🏜 Streamlit")
format_type = st.selectbox('Choose your OpenAI magician 😉',["ChatGPT","DALL-E 2"])



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
                # Upload an image and set some options for demo purposes
                st.header("Cropper Demo")
                img_file = bg_image
                box_color = '#000000'
                aspect_choice = st.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
                aspect_dict = {
                    "1:1": (1, 1),
                    "16:9": (16, 9),
                    "4:3": (4, 3),
                    "2:3": (2, 3),
                    "Free": None
                }
                aspect_ratio = aspect_dict[aspect_choice]

                if img_file:
                    img = Image.open(img_file)
                    # Get a cropped image from the frontend
                    cropped_img = st_cropper(img, realtime_update=True, box_color=box_color,
                                                aspect_ratio=aspect_ratio)
                    
                    # Manipulate cropped image at will
                    st.write("Preview")
                    _ = cropped_img.thumbnail((150,150))
                    st.image(cropped_img)
                    bg_image = cropped_img 

            # MASK
            mask_image = st.checkbox("mask image")
            if mask_image:
                # Specify canvas parameters in application
                # drawing_mode = st.sidebar.selectbox(
                #     "Drawing tool:",
                #     ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
                # )
                drawing_mode = "freedraw"
                stroke_width = st.slider("Stroke width: ", 1, 25, 20)
                if drawing_mode == 'point':
                    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
                # stroke_color = st.sidebar.color_picker("Stroke color hex: ")
                stroke_color = '#000000'
                bg_color =  "#eee"
                # bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
                # Resize the background image
                if Crop_image:
                    bg_image_ = bg_image
                else:
                    bg_image_ = Image.open(bg_image)
                width = 512
                # if bg_image_.size[0]>bg_image_.size[1]:
                #     width = bg_image_.size[0]
                # else:
                #     width = bg_image_.size[1]

                bg_image_resize = bg_image_.resize(size= (width,width))
                # alpha1 = bg_image_resize[:,:,3]
                # st.image(alpha1)
                # Create a new image with the same size as the background image
                # new_image = Image.new('RGBA', bg_image_resize.size)
                # # Paste the background image on the new image
                # new_image.paste(bg_image_resize, (0,0))
                new_image = bg_image_resize
                # st.write(bg_image_resize)
                # bg_image_ = Image.open(bg_image)
                # st.write(bg_image.size)
                # width, height = bg_image_.size[0], bg_image_.size[0]
                # bg_image_ = bg_image_.resize((width, height))
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

                    # Convert the image to a BytesIO object
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
                    # out = Image.convert("RGBA", mask_img)  
                    # st.write(out)
                # if canvas_result.json_data is not None:
                #     objects = pd.json_normalize(canvas_result.json_data["objects"])
                #     for col in objects.select_dtypes(include=["object"]).columns:
                #         objects[col] = objects[col].astype("str")
                #     st.dataframe(objects)
            

    variation_img = st.checkbox("Get variation of image")
    if variation_img and (not upload_img ):
        st.info("upload image first!!")
    input_text = st.text_area("Please enter text here... 🙋",height=50)
    image_button = st.button("Generate Image 🚀")
    if image_button and variation_img:
        # Read the image file from disk and resize it
        image = Image.open(original_upload_img)
        width, height = 512, 512
        image_r = image.resize((width, height))
        # Convert the image to a BytesIO object
        byte_stream = BytesIO()
        image_r.save(byte_stream, format='PNG')
        byte_array = byte_stream.getvalue()
        variation_url = openai_variation(byte_array)
        col1,col2,col3 = st.columns(3)
        col1.image(variation_url[0]['url'], caption='Generated by OpenAI Variation') 
        col2.image(variation_url[1]['url'], caption='Generated by OpenAI Variation') 
        col3.image(variation_url[2]['url'], caption='Generated by OpenAI Variation') 
    if not variation_img:

        if image_button and input_text.strip() != "":
            with st.spinner("Loading...💫"):
                # Convert the image to a BytesIO object

                # mask_img_image = Image.fromarray(np.uint8(cm.gist_earth(mask_img)*255))
                # out = Image.convert("RGBA", mask_img)  
                # st.write(out)
                # byte_stream_ = BytesIO()
                # im = Image.fromarray(mask_img, mode="RGBA")
                # st.write(im)
                # im.save(byte_stream_, format='PNG')
                # byte_array_mask = byte_stream_.getvalue()
                if mask_image:
                    response = openai.Image.create_edit(
                    image=byte_array_img,
                    mask= byte_array_mask,
                    prompt=input_text,
                    n=1,
                    size="512x512"
                    )
                    image_url = response['data'][0]['url']
                    st.image(image_url, caption='Generated by OpenAI')
                else :
                    image_url = openai_image(input_text)
                    st.image(image_url, caption='Generated by OpenAI')   
        else:
            st.warning("Please enter something! ⚠")
