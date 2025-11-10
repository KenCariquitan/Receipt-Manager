import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:supabase_flutter/supabase_flutter.dart';

import 'config.dart';
import 'models.dart';

class ApiClient {
  String get _base => AppConfig.apiBaseUrl;

  /// Build Authorization header from Supabase session.
  Future<Map<String, String>> _authHeaders() async {
    final session = Supabase.instance.client.auth.currentSession;
    final token = session?.accessToken;
    return {
      if (token != null && token.isNotEmpty) 'Authorization': 'Bearer $token',
    };
  }

  /// Optional shared decoder with simple error handling.
  T _decodeJson<T>(http.Response r) {
    if (r.statusCode >= 200 && r.statusCode < 300) {
      return jsonDecode(r.body) as T;
    }
    // Surface backend errors to the UI
    throw Exception(
        'HTTP ${r.statusCode} ${r.reasonPhrase} — ${r.body.isNotEmpty ? r.body : 'no body'}');
  }

  Future<Map<String, dynamic>> health() async {
    final r = await http
        .get(Uri.parse('$_base/health'), headers: await _authHeaders())
        .timeout(const Duration(seconds: 15));
    return _decodeJson<Map<String, dynamic>>(r);
  }

  Future<ReceiptJobStatus> uploadReceipt(MultipartFile file) async {
    final uri = Uri.parse('$_base/upload_receipt');
    final req = http.MultipartRequest('POST', uri)
      ..headers.addAll(await _authHeaders())
      ..files.add(file);

    final streamed = await req.send();
    final body = await streamed.stream.bytesToString();

    if (streamed.statusCode < 200 || streamed.statusCode >= 300) {
      throw Exception('HTTP ${streamed.statusCode} on /upload_receipt - $body');
    }
    final json = jsonDecode(body) as Map<String, dynamic>;
    return ReceiptJobStatus.fromJson(json);
  }

  Future<ReceiptJobStatus> jobStatus(String jobId) async {
    final uri = Uri.parse('$_base/jobs/$jobId');
    final r = await http
        .get(uri, headers: await _authHeaders())
        .timeout(const Duration(seconds: 15));
    final json = _decodeJson<Map<String, dynamic>>(r);
    return ReceiptJobStatus.fromJson(json);
  }

  Future<ReceiptJobStatus> waitForJob(
    String jobId, {
    Duration pollInterval = const Duration(seconds: 2),
    Duration timeout = const Duration(minutes: 3),
    void Function(ReceiptJobStatus status)? onUpdate,
  }) async {
    var status = await jobStatus(jobId);
    onUpdate?.call(status);

    final deadline = DateTime.now().add(timeout);
    while (!status.isFinal) {
      if (DateTime.now().isAfter(deadline)) {
        throw TimeoutException(
            'Job $jobId did not complete within ${timeout.inSeconds} seconds');
      }
      await Future.delayed(pollInterval);
      status = await jobStatus(jobId);
      onUpdate?.call(status);
    }
    return status;
  }

  Future<List<Receipt>> listReceipts({int limit = 50, int offset = 0}) async {
    final uri = Uri.parse('$_base/receipts?limit=$limit&offset=$offset');
    final r = await http
        .get(uri, headers: await _authHeaders())
        .timeout(const Duration(seconds: 20));
    final arr = _decodeJson<List>(r);
    return arr.map((e) => Receipt.fromJson(e as Map<String, dynamic>)).toList();
  }

  Future<SummaryStats> getSummary() async {
    final r = await http
        .get(Uri.parse('$_base/stats/summary'), headers: await _authHeaders())
        .timeout(const Duration(seconds: 15));
    return SummaryStats.fromJson(_decodeJson<Map<String, dynamic>>(r));
  }

  Future<List<Map<String, dynamic>>> statsByCategory() async {
    final r = await http
        .get(Uri.parse('$_base/stats/by_category'),
            headers: await _authHeaders())
        .timeout(const Duration(seconds: 15));
    final arr = _decodeJson<List>(r);
    return arr.cast<Map<String, dynamic>>();
  }

  Future<List<MerchantStat>> topMerchants({int limit = 5}) async {
    final uri = Uri.parse('$_base/stats/top_merchants?limit=$limit');
    final r = await http
        .get(uri, headers: await _authHeaders())
        .timeout(const Duration(seconds: 15));
    final arr = _decodeJson<List>(r);
    return arr
        .map((e) => MerchantStat.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<List<WeekdayStat>> weekdaySpend() async {
    final r = await http
        .get(Uri.parse('$_base/stats/weekday_spend'),
            headers: await _authHeaders())
        .timeout(const Duration(seconds: 15));
    final arr = _decodeJson<List>(r);
    return arr
        .map((e) => WeekdayStat.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<List<RollingStat>> rolling30DaySpend() async {
    final r = await http
        .get(Uri.parse('$_base/stats/rolling_30'),
            headers: await _authHeaders())
        .timeout(const Duration(seconds: 15));
    final arr = _decodeJson<List>(r);
    return arr
        .map((e) => RollingStat.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<List<Receipt>> lowConfidenceReceipts({
    double threshold = 0.6,
    int limit = 50,
  }) async {
    final uri = Uri.parse(
        '$_base/receipts/low_confidence?threshold=$threshold&limit=$limit');
    final r = await http
        .get(uri, headers: await _authHeaders())
        .timeout(const Duration(seconds: 20));
    final arr = _decodeJson<List>(r);
    return arr
        .map((e) => Receipt.fromJson(e as Map<String, dynamic>))
        .toList();
  }

  Future<bool> sendFeedback({
    required String text,
    required String trueLabel,
  }) async {
    final uri = Uri.parse('$_base/feedback');
    final r = await http.post(uri, headers: await _authHeaders(), body: {
      'text': text,
      'true_label': trueLabel
    }).timeout(const Duration(seconds: 20));
    if (r.statusCode == 200) {
      final j = jsonDecode(r.body) as Map<String, dynamic>;
      return (j['ok'] == true);
    }
    // bubble up errors (401/500) to caller
    throw Exception(
        'HTTP ${r.statusCode} ${r.reasonPhrase} — ${r.body.isNotEmpty ? r.body : 'no body'}');
  }

  Future<bool> updateReceipt({
    required String id,
    String? store,
    String? date, // ISO string
    double? total,
    String? category,
  }) async {
    final uri = Uri.parse('$_base/receipts/$id');
    final payload = <String, dynamic>{};
    if (store != null) payload['store'] = store;
    if (date != null) payload['date'] = date;
    if (total != null) payload['total'] = total;
    if (category != null) payload['category'] = category;

    final r = await http
        .patch(uri,
            headers: {
              'Content-Type': 'application/json',
              ...await _authHeaders(),
            },
            body: jsonEncode(payload))
        .timeout(const Duration(seconds: 20));

    if (r.statusCode == 200) {
      final j = jsonDecode(r.body) as Map<String, dynamic>;
      return j['ok'] == true;
    }
    throw Exception(
        'HTTP ${r.statusCode} ${r.reasonPhrase} — ${r.body.isNotEmpty ? r.body : 'no body'}');
  }
}
