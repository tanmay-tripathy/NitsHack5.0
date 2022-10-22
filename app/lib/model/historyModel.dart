class HistoryModel {
  String userID;
  double weight;
  String dustbinID;
  int? biodegradable;
  int? nonbiodegradable;
  int latitude;
  int logitude;
  int seconds;

  HistoryModel({
    required this.userID,
    required this.seconds,
    required this.weight,
    required this.dustbinID,
    required this.biodegradable,
    required this.nonbiodegradable,
    required this.latitude,
    required this.logitude,
  });

  factory HistoryModel.fromJson(Map<String, dynamic> json) {
    return HistoryModel(
        userID: json['userID'],
        seconds: json['timestamp']['_seconds'],
        weight: json['weight'],
        dustbinID: json['dustbinID'],
        biodegradable: json['composition']['biodegradable'] ?? 0,
        nonbiodegradable: json['composition']['nonbiodegradable'] ?? 0,
        latitude: json['location']['_latitude'],
        logitude: json['location']['_longitude']);
  }
}
