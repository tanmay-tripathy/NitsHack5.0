const db = require('../configs/firebaseConfig');

module.exports = {
    get : async (req,res) => {
        const docs = await db.collection('history')
            .where('dusbinID', '==' , req.params.dustbinID)
            .orderBy('timestamp','desc')
            .limit(1)
            .get();
        
        let objects = [];
        docs.forEach(elem => objects.push(elem.data()));
        res.status(200).send(objects);
    },

    all : async (req,res) => {
        const response = await db.collection('dustbinData').get();
        let dustbinIDs = [];
        response.forEach(elem => dustbinIDs.push({id: elem.id, data:elem.data()}));
        
        let composition = {
            biodegradable : 0,
            nonbiodegradable : {
                bottle: 0,
                glass: 0,
                metal: 0,
                plastic: 0    
            }
        };

        let weight = 0;

        dustbinIDs.forEach( async (elem) => {
            const docs = await db.collection('history')
                .where('dusbinID', '==' , elem.id)
                .orderBy('timestamp','desc').limit(1)
                .get();
            
            let object = [];
            docs.forEach(elem=> object.push(elem.data()));
            weight += object[0].weight;
            
            if(object[0].composition.biodegradable !== null) {
                composition.biodegradable += (object.biodegradable)*(object.weight)/100;
            }

            if(object[0].composition.nonbiodegradable !== null) {
                composition.nonbiodegradable.bottle += (object[0].nonbiodegradable.bottle)*(object[0].weight)/100;
                composition.nonbiodegradable.plastic += (object[0].nonbiodegradable.plastic)*(object[0].weight)/100;
                composition.nonbiodegradable.glass += (object[0].nonbiodegradable.glass)*(object[0].weight)/100;
                composition.nonbiodegradable.metal += (object[0].nonbiodegradable.metal)*(object[0].weight)/100;
            }
        });

        composition.nonbiodegradable.bottle /= weight/100;
        composition.nonbiodegradable.plastic /= weight/100;
        composition.nonbiodegradable.metal /= weight/100;
        composition.nonbiodegradable.glass /= weight/100;
        
        res.status(200).send(composition);
    }
}
