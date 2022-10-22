const express = require('express');
const router = express.Router();
const controller = require('../controllers/client.controller');

router.get('/:userID',controller.all);
router.post('/',controller.send);

module.exports = router;