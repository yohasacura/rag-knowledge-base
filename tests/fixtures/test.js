/**
 * Example JavaScript module for parser testing.
 * @module test
 */

function greet(name) {
    return `Hello, ${name}!`;
}

const API_URL = "https://api.example.com";
module.exports = { greet, API_URL };
