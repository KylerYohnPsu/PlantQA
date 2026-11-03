import * as general from "./general.js";
import * as posting from "./fast_posting.js";
import {HISTORY} from "./history.js";

let STATUS_SOCKET= null;
let ANSWER_SOCKET= null;

let ANSWER_RECONNECT_DELAY= 1000;
let ANSWER_RECONNECT_TIMER= null;

let STATUS_RECONNECT_DELAY= 1000;
let STATUS_RECONNECT_TIMER= null;

const MAX_RECONNECT_DELAY= 5000;

document.addEventListener("DOMContentLoaded", initialize_websockets);

function initialize_websockets() {
    initialize_status_socket();
    initialize_answer_socket();
}

/****** STATUS SOCKET FUNCTIONS *****/

function initialize_status_socket() {
    if (status_socket_ready())
        return;

    STATUS_SOCKET = new WebSocket("ws://127.0.0.1:8000/status");

    STATUS_SOCKET.onopen = status_socket_open;
    STATUS_SOCKET.onmessage = status_socket_receive;
    STATUS_SOCKET.onerror = status_socket_error;
    STATUS_SOCKET.onclose = status_socket_close
}

async function status_socket_receive(event) {
    const status_data= JSON.parse(event.data);
    const status= status_data.status;

    general.update_status_text(status);
}

function status_socket_ready() {
    if (STATUS_SOCKET &&
        (STATUS_SOCKET.readyState === WebSocket.OPEN ||
        STATUS_SOCKET.readyState === WebSocket.CONNECTING)) {
        return true;
    }
    return false;
}

function status_socket_open() {
    posting.py_log("STATUS SOCKET CONNECTED");
    STATUS_RECONNECT_DELAY= 1000;
    clearTimeout(STATUS_RECONNECT_TIMER);
    STATUS_RECONNECT_TIMER= null;
}

function status_socket_error(error) {
    posting.py_log(`STATUS SOCKET ERROR: ${error}`);
    STATUS_SOCKET.close();
}

function status_socket_close() {
    posting.py_log("STATUS SOCKET DISCONNECTED!");
    status_socket_reconnect();
}

function status_socket_reconnect() {
    if (STATUS_RECONNECT_TIMER !== null)
        return;

    posting.py_log("Attempting to reconnect STATUS SOCKET");

    STATUS_RECONNECT_TIMER= setTimeout(() => {
        STATUS_RECONNECT_TIMER= null;

        initialize_status_socket();

        STATUS_RECONNECT_DELAY= STATUS_RECONNECT_DELAY * 2;
        if (STATUS_RECONNECT_DELAY > MAX_RECONNECT_DELAY)
            STATUS_RECONNECT_DELAY= MAX_RECONNECT_DELAY;

    }, STATUS_RECONNECT_DELAY);
}

/***** ANSWER SOCKET FUNCTIONS *******/

function initialize_answer_socket() {
    if (answer_socket_ready())
        return;

    ANSWER_SOCKET = new WebSocket("ws://127.0.0.1:8000/answer");

    ANSWER_SOCKET.onopen = answer_socket_open;
    ANSWER_SOCKET.onmessage = answer_socket_receive;
    ANSWER_SOCKET.onerror = answer_socket_error;
    ANSWER_SOCKET.onclose = answer_socket_close;
}

async function answer_socket_receive(event) {
    const answer_data= JSON.parse(event.data);
    const answer= answer_data.answer;
    const confidence= answer_data.confidence;
    const job_id= answer_data.job_id;
    const classification= answer_data.classification;
    const top_k= answer_data.top_k;

    posting.py_log(classification);
    posting.py_log(top_k);

    general.set_answer(answer);
    general.set_confidence(top_k);
    HISTORY.update_answer(job_id, answer, confidence);
}

function answer_socket_ready() {
    if (ANSWER_SOCKET &&
        (ANSWER_SOCKET.readyState === WebSocket.OPEN ||
        ANSWER_SOCKET.readyState === WebSocket.CONNECTING)) {
        return true;
    }
    return false;
}

function answer_socket_open() {
    posting.py_log("ANSWER SOCKET CONNECTED");
    ANSWER_RECONNECT_DELAY= 1000;
    clearTimeout(ANSWER_RECONNECT_TIMER);
    ANSWER_RECONNECT_TIMER= null;
}

function answer_socket_error(error) {
    posting.py_log(`ANSWER SOCKET ERROR: ${error}`);
    ANSWER_SOCKET.close();
}

function answer_socket_close() {
    posting.py_log("ANSWER SOCKET DISCONNECTED!");
    answer_socket_reconnect();
}

function answer_socket_reconnect() {
    if (ANSWER_RECONNECT_TIMER !== null)
        return;

    posting.py_log("Attempting to reconnect ANSWER SOCKET");

    ANSWER_RECONNECT_TIMER= setTimeout(() => {
        ANSWER_RECONNECT_TIMER= null;

        initialize_answer_socket();

        ANSWER_RECONNECT_DELAY= ANSWER_RECONNECT_DELAY * 2;
        if (ANSWER_RECONNECT_DELAY > MAX_RECONNECT_DELAY)
            ANSWER_RECONNECT_DELAY= MAX_RECONNECT_DELAY;

    }, ANSWER_RECONNECT_DELAY);
}