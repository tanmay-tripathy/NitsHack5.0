const express = require('express');
const app = express();
const cors = require('cors');
const dotenv = require('dotenv');
dotenv.config();

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({extended: false}));


const user = require('./routes/client.router');

app.use('/user',user);


const PORT = process.env.PORT || 3000;
app.listen(PORT,()=>{console.log(`listening on http://localhost:${PORT}`)});