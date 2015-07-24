db.Dedicated_Page.aggregate([
    {
        '$match':
            {
                '_id': idRD('2015/07/10', 30),
                "service": "buy",
                "service_type": "projects",
                "id":
                {"$in": [1205]},
                "action": "landing"
            }
    },
    {'$project':
     {
         'datetime': {$add: [new Date(0), '$timeStamp']},
         datetime:1,
         _id:0
     }
     },
    {'$project':
     {
         date: {$dateToString: {format: "%Y-%m-%d", date: "$datetime"}},
     }
     },
    {
        '$group':
            {
                '_id': '$date',
                'count': {'$sum': 1}
            }
    }
])


db.Dedicated_Page.aggregate([
    {
        '$match':
            {
                '_id': idRD('2015/07/10', 30),
                "service": "buy",
                "service_type": "projects",
                "polygon_id": '0136de9c12c05fc6a7a9',
                "action": "landing"
            }
    },
    {'$project':
     {
         'datetime': {$add: [new Date(0), '$timeStamp']},
         datetime:1,
         _id:0
     }
     },
    {'$project':
     {
         date: {$dateToString: {format: "%Y-%m-%d", date: "$datetime"}},
     }
     },
    {
        '$group':
            {
                '_id': '$date',
                'count': {'$sum': 1}
            }
    }
])
