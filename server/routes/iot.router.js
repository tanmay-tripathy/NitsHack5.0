const express = require('express');
const router = express.Router();
const controller = require('../controllers/iot.controller')

router.post('/',controller.send)

module.exports = router;