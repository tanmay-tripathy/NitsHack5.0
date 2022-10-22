const db = require('../configs/firebaseConfig');

module.exports = {
    
    all: async (req,res)=>{
        try{
            const docs = await db.collection('history')
                .where('userID','==',req.params.userID)
                .orderBy('timestamp','desc')
                .get();
        
            let objects = [];
            docs.forEach(elem=>objects.push(elem.data()));
            res.status(200).send(objects);

        } catch (err) {
            console.log(err);
            res.status(503).send(err);
        }
    },

    send: async (req,res)=>{
        try{
            console.log(req.body);
            const docs = await db.collection('history')
                .where('dustbinID','==',req.body.dustbinID)
                .where('userID','==',null)
                .orderBy('timestamp','desc')
                .limit(1)
                .get();

            let object = [];
            docs.forEach(elem => object.push({id:elem.id ,data: elem.data()}));
            console.log(object);

            if(object.length == 0) res.status(500).send('No disposals triggered yet');

            object[0].data.userID = req.body.userID;
            await db.collection('history').doc(object[0].id).set({userID : req.body.userID},{merge: true});
            const response = await db.collection(req.body.dustbinID).doc().set(object[0].data);
            res.status(200).send(response);

        } catch (err) {
            console.log(err);
            res.status(500).send(err);
        }
    }
}
