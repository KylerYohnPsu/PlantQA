
async function py_log(data) {
    /*
    Simple function that sends a string to the python side
    that way the python side logs it to the terminal
    */
    const log_link="http://127.0.0.1:8000/log";
    const log_data= new FormData();
    log_data.append("log", data);
    const log_content= {method: "POST", body: log_data};
    
    try {
        const response= await fetch(log_link, log_content);
        if (!response.ok)
            throw new Error(`HTTP ERROR: ${response.status}`);
    } catch (error) {
        console.error(error);
    }
}

async function send_question_prompt(question, image) {
    /*
    Send a question/image pair to the backend for processing
    */
    const ask_link= "http://127.0.0.1:8000/ask"
    const ask_data= new FormData();
    ask_data.append("image", image);
    ask_data.append("question", question);
    const ask_content= {method: "POST", body: ask_data};


    try {
        console.log("Attempting to send question...")
        const response= await fetch(ask_link, ask_content);

        if (!response.ok)
            throw new Error(`HTTP ERROR: ${response.status}`);

        const response_data= await response.json();

        return response_data;
    } catch (error) {
        console.error(error);
        return error
    }

}

