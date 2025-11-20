import 'package:flutter/material.dart';
import '../models.dart';

class ReceiptCard extends StatelessWidget {
  final UploadResult res;
  final String? selectedCategory;
  final ValueChanged<String?>? onCategoryChanged;
  final TextEditingController storeController;
  final TextEditingController totalController;
  final TextEditingController dateController;

  const ReceiptCard({
    super.key,
    required this.res,
    this.selectedCategory,
    this.onCategoryChanged,
    required this.storeController,
    required this.totalController,
    required this.dateController,
  });

  static const cats = [
    'Utilities',
    'Food',
    'Groceries',
    'Transportation',
    'Health & Wellness',
    'Others'
  ];

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ValueListenableBuilder<TextEditingValue>(
              valueListenable: storeController,
              builder: (context, value, _) {
                final storeName =
                    value.text.isNotEmpty ? value.text : (res.store ?? 'Unknown Store');
                return Text(storeName, style: textTheme.titleLarge);
              },
            ),
            const SizedBox(height: 12),
            TextField(
              controller: storeController,
              decoration: const InputDecoration(
                labelText: 'Store / Merchant',
                helperText: 'Edit if the merchant name is incorrect.',
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: totalController,
              keyboardType: const TextInputType.numberWithOptions(decimal: true),
              decoration: InputDecoration(
                labelText: 'Total Amount',
                helperText: res.total != null
                    ? 'Detected total: ${res.total!.toStringAsFixed(2)}'
                    : 'Enter the total from the receipt.',
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: dateController,
              keyboardType: TextInputType.datetime,
              decoration: InputDecoration(
                labelText: 'Receipt Date (YYYY-MM-DD)',
                helperText: res.date != null
                    ? 'Detected date: ${res.date}'
                    : 'Enter an ISO date if missing.',
              ),
            ),
            const SizedBox(height: 12),
            Text('Predicted: ${res.category ?? 'No model'}'),
            // Debug info (confidence) removed for clean UI; restore if needed.
            // if (res.confidence != null)
            //   Text('Confidence: ${(res.confidence! * 100).toStringAsFixed(1)}%'),
            const SizedBox(height: 8),
            DropdownButtonFormField<String>(
              initialValue: selectedCategory ?? res.category ?? 'Others',
              items: cats.map((c) => DropdownMenuItem(value: c, child: Text(c))).toList(),
              onChanged: onCategoryChanged,
              decoration: const InputDecoration(labelText: 'Correct / Confirm Category'),
            ),
            const SizedBox(height: 8),
            ExpansionTile(
              title: const Text('OCR Text'),
              children: [
                SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Text(res.text),
                ),
              ],
            ),
            // Debug engine breakdown hidden for production use. Uncomment to inspect.
            // const Divider(height: 20),
            // Text(
            //   'Engines - YOLO: ${res.yoloUsed} | OCR.space: ${res.ocrSpaceUsed} | Final OCR: ${res.ocrSourceLabel ?? res.ocrSource ?? 'n/a'}',
            //   style: Theme.of(context).textTheme.bodySmall,
            // ),
          ],
        ),
      ),
    );
  }
}
