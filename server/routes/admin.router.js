const router = express.Router();
const controller = require('../controllers/admin.controller')

router.get('/:dustbinID',controller.get);
router.get('/',controller.all)

module.exports = router;