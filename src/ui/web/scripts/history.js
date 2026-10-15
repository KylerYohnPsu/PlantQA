class History extends EventTarget {
    constructor() {
        super();
        this._entries= new Map();
        this._table= document.getElementById("history-table");
        this._tableContents= document.getElementById("history-table-contents");
    }//constructor

    add(job_id, question, image) {
        const entry= {
            job_id,
            question,
            image,
            confidence: "--",
            image_url: URL.createObjectURL(image),
            answer: "<PENDING>",
            status: "pending",
        };//entry

        this._entries.set(job_id, entry);
        
        this.populate_table();
    }//add

    update_answer(job_id, answer, confidence) {
        const entry= this._entries.get(job_id);
        if (!entry)
            return;

        console.log(`Updating answer for id: ${job_id}`);
        const updates= {answer, status: "answered", confidence}
        Object.assign(entry, updates);
        this.populate_table();
    }

    remove_entry(job_id) {
        const entry= this._entries.get(job_id);
        if (!entry)
            return;

        URL.revokeObjectURL(entry.image_url);
        this._entries.delete(job_id);
    }

    get_entry(job_id) {return this._entries.get(job_id);}
    get_entries() {return [...this._entries.values()];}

    populate_table() {
        this._tableContents.innerHTML= "";

        console.log(this._entries);

        for (const [job_id, data] of this._entries) {
            const row= document.createElement('tr');
            row.dataset.id= job_id;
            const row_cells= [data.image_url, data.question, data.answer, data.confidence, job_id];
            
            row_cells.forEach((entry, i) => {
                const cell= document.createElement('td');
                
                if (i === 0) {
                    const img= document.createElement('img');
                    img.src= entry;
                    cell.appendChild(img);
                } else {
                    cell.textContent= entry;
                }

                row.appendChild(cell);
            });

            this._tableContents.prepend(row);
        }
    }
}//class

export const HISTORY= new History();