import 'dart:ffi';

class HistoryModel {
  String userID;
  Map<String, dynamic> timestamp;
  Float weight;
  String dusbinID;
  Map<String, dynamic> composition;
  Map<String, dynamic> location;

  HistoryModel(
      {required this.userID,
      required this.timestamp,
      required this.weight,
      required this.dusbinID,
      required this.composition,
      required this.location});

  factory HistoryModel.fromJson(Map<String, dynamic> json) {
    return HistoryModel(
        userID: json['userID'],
        timestamp: json['timestamp'],
        weight: json['weight'],
        dusbinID: json['dusbinID'],
        composition: json['composition'],
        location: json['location']);
  }
}
