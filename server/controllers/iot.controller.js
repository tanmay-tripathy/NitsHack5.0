const db = require('../configs/firebaseConfig');

module.exports = {
    send : async (res,req) => {
        try{
            const response = await db.collection('history').doc().set({
                userID: null,
                dustbinID: req.body.dustbinID,
                location: req.body.location,
                weight: req.body.weight,
                composition: req.body.composition,
                timestamp: req.body.timestamp
            });
            res.status(200).send(response);
        } catch (err) {
            console.log(err);
            res.status(500).send(err);
        }
    }
}