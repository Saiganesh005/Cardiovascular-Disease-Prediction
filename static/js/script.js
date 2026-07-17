/*
==========================================================
Project : Cardiovascular Disease Prediction
File    : script.js
Author  : Robbi Sai Ganesh Devi Prasad

Description
-----------
JavaScript utilities for the Flask application.

Features
--------
1. Form Validation
2. Loading Button
3. BMI Calculator
4. Reset Confirmation
5. Number Validation
6. Progress Animation
7. Print Report
==========================================================
*/

"use strict";

/* ==========================================================
   DOM Loaded
========================================================== */

document.addEventListener("DOMContentLoaded", () => {

    initialize();

});

/* ==========================================================
   Initialize
========================================================== */

function initialize(){

    validateNumbers();

    calculateBMI();

    animateProgressBars();

}

/* ==========================================================
   Loading Button
========================================================== */

function showLoading(button){

    button.disabled = true;

    button.innerHTML =
        '<span class="spinner-border spinner-border-sm"></span> Predicting...';

}

/* ==========================================================
   Form Validation
========================================================== */

function validateForm(){

    const age = Number(document.getElementsByName("age")[0].value);

    const height = Number(document.getElementsByName("height")[0].value);

    const weight = Number(document.getElementsByName("weight")[0].value);

    const ap_hi = Number(document.getElementsByName("ap_hi")[0].value);

    const ap_lo = Number(document.getElementsByName("ap_lo")[0].value);

    if(age <= 0){

        alert("Age must be greater than zero.");

        return false;

    }

    if(height <= 0){

        alert("Height must be greater than zero.");

        return false;

    }

    if(weight <= 0){

        alert("Weight must be greater than zero.");

        return false;

    }

    if(ap_hi <= ap_lo){

        alert("Systolic BP must be greater than Diastolic BP.");

        return false;

    }

    return true;

}

/* ==========================================================
   BMI Calculator
========================================================== */

function calculateBMI(){

    const heightField = document.getElementsByName("height")[0];

    const weightField = document.getElementsByName("weight")[0];

    const bmiOutput = document.getElementById("bmi");

    if(!heightField || !weightField || !bmiOutput){

        return;

    }

    function updateBMI(){

        const h = Number(heightField.value);

        const w = Number(weightField.value);

        if(h > 0 && w > 0){

            const bmi = w / Math.pow(h / 100, 2);

            bmiOutput.innerHTML = bmi.toFixed(2);

        }
        else{

            bmiOutput.innerHTML = "--";

        }

    }

    heightField.addEventListener("keyup", updateBMI);

    weightField.addEventListener("keyup", updateBMI);

    heightField.addEventListener("change", updateBMI);

    weightField.addEventListener("change", updateBMI);

}

/* ==========================================================
   Reset Confirmation
========================================================== */

function confirmReset(){

    return confirm(

        "Reset all patient information?"

    );

}

/* ==========================================================
   Number Validation
========================================================== */

function validateNumbers(){

    const numbers = document.querySelectorAll("input[type='number']");

    numbers.forEach(field => {

        field.addEventListener("keypress", function(e){

            if(e.key === "-"){

                e.preventDefault();

            }

        });

    });

}

/* ==========================================================
   Progress Bar Animation
========================================================== */

function animateProgressBars(){

    const progressBars = document.querySelectorAll(".progress-bar");

    progressBars.forEach(bar => {

        const width = bar.style.width;

        bar.style.width = "0%";

        setTimeout(() => {

            bar.style.width = width;

        },300);

    });

}

/* ==========================================================
   Print Report
========================================================== */

function printReport(){

    window.print();

}

/* ==========================================================
   Download Report
========================================================== */

function downloadReport(){

    window.print();

}

/* ==========================================================
   Auto Scroll
========================================================== */

function scrollTopPage(){

    window.scrollTo({

        top:0,

        behavior:"smooth"

    });

}

/* ==========================================================
   Show Alert
========================================================== */

function showMessage(message,type="success"){

    const alertBox = document.createElement("div");

    alertBox.className =
        `alert alert-${type} position-fixed top-0 end-0 m-4`;

    alertBox.style.zIndex = "9999";

    alertBox.innerHTML = message;

    document.body.appendChild(alertBox);

    setTimeout(()=>{

        alertBox.remove();

    },3000);

}

/* ==========================================================
   Copy Result
========================================================== */

function copyPrediction(){

    const text = document.body.innerText;

    navigator.clipboard.writeText(text);

    showMessage(

        "Prediction copied successfully."

    );

}

/* ==========================================================
   Dark Mode
========================================================== */

function toggleDarkMode(){

    document.body.classList.toggle("bg-dark");

    document.body.classList.toggle("text-white");

}

/* ==========================================================
   Form Submit
========================================================== */

const form = document.querySelector("form");

if(form){

    form.addEventListener("submit",function(e){

        if(!validateForm()){

            e.preventDefault();

            return;

        }

        const button = this.querySelector("button[type='submit']");

        if(button){

            showLoading(button);

        }

    });

}

/* ==========================================================
   Finished
========================================================== */

console.log(

    "Cardio Predict JavaScript Loaded Successfully."

);
