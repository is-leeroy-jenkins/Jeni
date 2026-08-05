document.addEventListener("DOMContentLoaded", function () {
    const tables = document.querySelectorAll("table");

    tables.forEach(function (table) {
        if (!table.parentElement.classList.contains("md-typeset__scrollwrap")) {
            table.setAttribute("tabindex", "0");
        }
    });
});
