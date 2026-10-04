document.addEventListener("DOMContentLoaded", initialize);

function initialize() {
    const image_upload= document.getElementById("image-upload");
    const prompt_button= document.getElementById("prompt-button");

    image_upload.addEventListener("change", upload_image);
    prompt_button.addEventListener("click", ask_question);

    py_log("UI Initialized!")
}

function upload_image() {
    const upload_label= document.getElementById("image-upload-content");
    const image_preview= document.getElementById("image-preview");
    const image_preview_container= document.getElementById("image-preview-container");

    const file= this.files[0];
    if (!file) 
        return;
    
    console.log("Loading image...")
    image_preview.src= URL.createObjectURL(file);
    
    upload_label.style.display= "none";
    image_preview_container.style.display= "flex";
}//_upload_image

function get_question() {
    const question_input= document.getElementById("question-input");
    const question= question_input.value;
    return question;
}

function get_image() {
    const image_upload= document.getElementById("image-upload");
    const image= image_upload.files[0];
    return image
}

async function ask_question() {
    const question= get_question();
    if (question === "") {
        console.log("No question input");
        return
    }

    const image= get_image();
    if (!image) {
        console.log("No image input")
        return;
    }

    console.log("Question: ", question);
    console.log("Image: ", image);

    const response= await send_question_prompt(question, image);

    console.log("VQA Job ID: ", response.job_id);
    
}

function update_status_text(status) {
    const status_text= document.getElementById("status-text");
    if (!status_text)
        return;

    status_text.textContent= status.toUpperCase();

    switch (status.toLowerCase()) {
        case "ready": 
            status_text.style.color= "#07f045";
            status_text.style.fontSize= "clamp(4px, 4vw, 100px)";
            break;
        case "processing":
            status_text.style.color= "yellow";
            break;
        case "error":
            status_text.style.color= "red";
            break;
        default:
            status_text.style.color= "white";
            break;
    }//switch
}