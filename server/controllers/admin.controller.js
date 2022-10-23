const db = require('../configs/firebaseConfig');

module.exports = {
    get : async (req,res) => {
        try {
            const docs = await db.collection('history')
                .where('dustbinID', '==' , req.params.dustbinID)
                .orderBy('timestamp','desc')
                .limit(1)
                .get();
        
            let objects = [];
            docs.forEach(elem => objects.push(elem.data()));
            res.status(200).send(objects[0]);
        } catch(err) {
            res.status(500).send(err);            
        }
    },

    all : async (req,res) => {
        let dustbinIDs = [];
        
        try {
            const response = await db.collection('dustbinData').get();
            response.forEach(elem => dustbinIDs.push({id: elem.id, data:elem.data()}));
        } catch(err) {
            return res.status(500).send(err);
        }
    
        let objects = [];
        for await (const elem of dustbinIDs) {
            const response = await db.collection('history')
                .where('dustbinID','==',elem.id)
                .orderBy('timestamp','desc').limit(1)
                .get();
            
            response.forEach(elem=>objects.push(elem.data()));
        }

        let weight = 0; 
        let biodegradable = 0;
        let bottle = 0;
        let metal = 0;
        let plastic = 0;
        let glass = 0;

        for (const elem of objects) {
            weight += elem.weight;

            if(elem.composition.biodegradable !== null) {
                console.log('bio not null');
                biodegradable += elem.composition.biodegradable;
            }
            
            if(elem.composition.nonbiodegradable !== null) {
                console.log('non bio null');    
                bottle += elem.composition.nonbiodegradable.bottle*elem.weight/100;
                metal += elem.composition.nonbiodegradable.metal*elem.weight/100;
                glass += elem.composition.nonbiodegradable.glass*elem.weight/100;
                plastic += elem.composition.nonbiodegradable.plastic*elem.weight/100;   
            }
        }

        biodegradable /= weight/100;
        bottle /= weight/100;
        metal /= weight/100;
        glass /= weight/100;
        plastic /= weight/100;

        res.status(200).json({
            allDustbins: objects,
            weight: weight,
            composition: {
                biodegradable: biodegradable,
                nonbiodegradable: {
                    bottle: bottle,
                    metal: metal,
                    plastic: plastic,
                    glass: glass         
                }
            }
        });
    }
}
