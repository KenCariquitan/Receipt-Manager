import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:http/http.dart';
import 'config.dart';
import 'models.dart';

class ApiClient {
  String get _base => AppConfig.apiBaseUrl;

  Future<Map<String, dynamic>> health() async {
    final r = await http.get(Uri.parse('$_base/health')).timeout(const Duration(seconds: 15));
    return jsonDecode(r.body) as Map<String, dynamic>;
  }

  Future<UploadResult> uploadReceipt(MultipartFile file) async {
    final uri = Uri.parse('$_base/upload_receipt');
    final req = http.MultipartRequest('POST', uri)..files.add(file);
    final streamed = await req.send();
    final body = await streamed.stream.bytesToString();
    final json = jsonDecode(body) as Map<String, dynamic>;
    return UploadResult.fromJson(json);
  }

  Future<List<Receipt>> listReceipts({int limit = 50, int offset = 0}) async {
    final uri = Uri.parse('$_base/receipts?limit=$limit&offset=$offset');
    final r = await http.get(uri).timeout(const Duration(seconds: 20));
    final arr = jsonDecode(r.body) as List;
    return arr.map((e) => Receipt.fromJson(e as Map<String, dynamic>)).toList();
  }

  Future<SummaryStats> getSummary() async {
    final r = await http.get(Uri.parse('$_base/stats/summary')).timeout(const Duration(seconds: 15));
    return SummaryStats.fromJson(jsonDecode(r.body) as Map<String, dynamic>);
  }

  Future<List<Map<String, dynamic>>> statsByCategory() async {
    final r = await http.get(Uri.parse('$_base/stats/by_category')).timeout(const Duration(seconds: 15));
    final arr = jsonDecode(r.body) as List;
    return arr.cast<Map<String, dynamic>>();
  }

  Future<bool> sendFeedback({required String text, required String trueLabel}) async {
    final uri = Uri.parse('$_base/feedback');
    final r = await http.post(uri, body: {
      'text': text,
      'true_label': trueLabel,
    }).timeout(const Duration(seconds: 20));
    if (r.statusCode == 200) {
      final j = jsonDecode(r.body) as Map<String, dynamic>;
      return (j['ok'] == true);
    }
    return false;
  }

  Future<bool> updateReceipt({
    required String id,
    String? store,
    String? date,
    double? total,
    String? category,
  }) async {
    final uri = Uri.parse('$_base/receipts/$id');
    final payload = <String, dynamic>{};
    if (store != null) payload['store'] = store;
    if (date != null) payload['date'] = date;
    if (total != null) payload['total'] = total;
    if (category != null) payload['category'] = category;

    final r = await http.patch(
      uri,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(payload),
    ).timeout(const Duration(seconds: 20));

    if (r.statusCode == 200) {
      final j = jsonDecode(r.body) as Map<String, dynamic>;
      return j['ok'] == true;
    }
    return false;
  }
}
