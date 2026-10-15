import {HISTORY} from "./history.js";
import * as posting from "./fast_posting.js";

document.addEventListener("DOMContentLoaded", initialize);

function initialize() {
    const image_upload= document.getElementById("image-upload");
    const prompt_button= document.getElementById("prompt-button");

    image_upload.addEventListener("change", upload_image);
    prompt_button.addEventListener("click", ask_question);

    posting.py_log("UI Initialized!")
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

export function get_question() {
    const question_input= document.getElementById("question-input");
    const question= question_input.value;
    return question;
}

export function get_image() {
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

    const response= await posting.send_question_prompt(question, image);
    
    HISTORY.add(response.job_id, question, image);

    posting.py_log(`Entry added to history for ID: ${response.job_id}`);
}

export function update_status_text(status) {
    const status_text= document.getElementById("status-text");
    const status_light= document.getElementById("status-light");
    if (!status_text)
        return;

    status_text.style.fontSize= "clamp(3px, 2.5vw, 80px)";

    status_text.textContent= status.toUpperCase();

    switch (status.toLowerCase()) {
        case "ready": 
            status_light.style.background= "#07f045";
            status_light.style.boxShadow= "0 0 6px #07f045";
            break;
        case "embedding":
        case "generating":
        case "predicting":
        case "preprocessing":
        case "processing":
            status_light.style.background= "yellow";
            status_light.style.boxShadow= "0 0 6px yellow";
            break;
        case "error":
            status_light.style.background= "red";
            status_light.style.boxShadow= "0 0 6px red";
            break;
        default:
            status_light.style.background= "white";
            status_light.style.boxShadow= "0 0 6px white";
            break;
    }//switch
}

export function set_answer(new_answer) {
    const placeholder= document.getElementById("answer-placeholder");
    const answer= document.getElementById("answer");

    placeholder.style.display= "none";
    answer.textContent= ""
    answer.style.display= "flex";

    let char_index= 0;
    const speed= 20;

    function type_next_character() {
        if (char_index < new_answer.length) {
            answer.textContent+= new_answer[char_index];
            char_index++;

            setTimeout(type_next_character, speed);
        }
    }

    type_next_character();
}

export function set_confidence(confidence) {
    //const confidence_text= document.getElementById("confidence");

}