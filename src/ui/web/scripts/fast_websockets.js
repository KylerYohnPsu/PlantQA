import * as general from "./general.js";
import * as posting from "./fast_posting.js";
import {HISTORY} from "./history.js";

let STATUS_SOCKET= null;
let ANSWER_SOCKET= null;

document.addEventListener("DOMContentLoaded", initialize_websockets);

function initialize_websockets() {
    initialize_status_socket();
    initialize_answer_socket();
}

function initialize_answer_socket() {
    STATUS_SOCKET=  new WebSocket("ws://127.0.0.1:8000/answer");
    STATUS_SOCKET.onopen= function() {};
    STATUS_SOCKET.onmessage= answer_socket_receive;
    STATUS_SOCKET.onerror= function(error) {posting.py_log(`STATUS SOCKET ERROR: ${error}`);};
    STATUS_SOCKET.onclose= function() {};
}

function initialize_status_socket() {
    ANSWER_SOCKET=  new WebSocket("ws://127.0.0.1:8000/status");
    ANSWER_SOCKET.onopen= function() {};
    ANSWER_SOCKET.onmessage= status_socket_receive;
    ANSWER_SOCKET.onerror= function(error) {posting.py_log(`ANSWER SOCKET ERROR: ${error}`);};
    ANSWER_SOCKET.onclose= function() {};
}

async function status_socket_receive(event) {
    const status_data= JSON.parse(event.data);
    const status= status_data.status;

    general.update_status_text(status);
}
    
async function answer_socket_receive(event) {
    const answer_data= JSON.parse(event.data);
    const answer= answer_data.answer;
    const confidence= answer_data.confidence;
    const job_id= answer_data.job_id;

    general.set_answer(answer_data.answer);
    general.set_confidence(answer_data.confidence);
    HISTORY.update_answer(job_id, answer, confidence);
    
    
}
