let STATUS_SOCKET= null;

document.addEventListener("DOMContentLoaded", initialize_websockets);

function initialize_websockets() {
    initialize_status_socket();
}

function initialize_status_socket() {
    STATUS_SOCKET=  new WebSocket("ws://127.0.0.1:8000/status");
    STATUS_SOCKET.onopen= status_socket_send_received;
    STATUS_SOCKET.onmessage= status_socket_receive;
    STATUS_SOCKET.onerror= function(error) {py_log(`STATUS SOCKET ERROR: ${error}`);};
    STATUS_SOCKET.onclose= function() {};
}

async function status_socket_send_received() {
    await STATUS_SOCKET.send("received");
}

async function status_socket_receive(event) {
    const status_data= JSON.parse(event.data);
    const status= status_data.status;

    update_status_text(status);

    console.log(status);

    await status_socket_send_received();
}
    
