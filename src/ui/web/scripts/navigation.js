const NAV_BUTTONS= document.querySelectorAll(".nav-item");
const PAGES= document.querySelectorAll(".page");

function change_page(page_id) {
    PAGES.forEach( 
        (page) => {
            page.classList.toggle("active", page.id === page_id);
        }
    );

    NAV_BUTTONS.forEach(
        (item) => {
            const is_active= item.dataset.page === page_id;

            item.classList.toggle("active", is_active);
            item.setAttribute("aria-current", is_active ? "page" : "false");
        }
    );
}//change page

NAV_BUTTONS.forEach(
    (item) => {
        item.addEventListener("click", () => {
            change_page(item.dataset.page);
        });
    }
);