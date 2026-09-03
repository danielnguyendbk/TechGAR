import { expect, test } from '@playwright/test';

test('F-A parked vehicle remains visible without an observation', async ({ page }) => {
  await page.goto('/monitor?fixture=parked-long');
  await page.waitForTimeout(400);
  await expect(page.getByLabel('Xe Global ID 17')).toHaveCount(1);
  await expect(page.getByLabel(/Xe Global ID 17, đã đỗ tại D09/)).toBeVisible();
});

test('F-B one-frame gaps keep one stable marker', async ({ page }) => {
  await page.goto('/?fixture=flicker-gap');
  await page.waitForTimeout(400);
  await expect(page.getByLabel('Xe Global ID 17')).toHaveCount(1);
});

test('F-C ghost is held briefly and then hidden', async ({ page }) => {
  await page.goto('/monitor?fixture=ghost');
  await expect(page.getByLabel('Xe Global ID 99')).toBeVisible();
  await expect(page.getByLabel('Xe Global ID 99')).toHaveCount(0, { timeout: 3_000 });
});

test('F-D driver page exposes only its session vehicle and confirmed route', async ({ page }) => {
  await page.goto('/?fixture=driver-isolation&session=S42');
  await expect(page.getByLabel('Xe Global ID 17')).toHaveCount(1);
  await expect(page.getByLabel('Xe Global ID 5')).toHaveCount(0);
  await expect(page.getByLabel('Xe Global ID 22')).toHaveCount(0);
  await expect(page.getByLabel('Tuyến đường đã xác nhận')).toBeVisible();
});

test('F-E parked session falls back to the owned slot center', async ({ page }) => {
  await page.goto('/?fixture=parked-fallback&session=S42');
  await expect(page.getByLabel(/Xe Global ID 17, đã đỗ tại B04/)).toBeVisible();
  await expect(page.getByText('Đã đỗ tại B04')).toBeVisible();
  await expect(page.getByRole('button', { name: /Bắt đầu chỉ đường ra/ })).toBeVisible();
});

test('F-F route is created only after explicit confirmation', async ({ page }) => {
  await page.goto('/?fixture=normal');
  await page.getByRole('button', { name: 'D01', exact: true }).click();
  await expect(page.getByLabel('Tuyến đường đã xác nhận')).toHaveCount(0);
  await page.getByRole('button', { name: 'Chọn lại' }).click();
  await expect(page.getByLabel('Tuyến đường đã xác nhận')).toHaveCount(0);
  await page.getByRole('button', { name: 'D01', exact: true }).click();
  await page.getByRole('button', { name: /Xác nhận ô D01/ }).click();
  await expect(page.getByLabel('Tuyến đường đã xác nhận')).toBeVisible();
});

test('F-G off-route warns without silently replacing the confirmed route', async ({ page }) => {
  await page.goto('/?fixture=off-route&session=S42');
  const route = page.getByLabel('Tuyến đường đã xác nhận').locator('polyline').first();
  await expect(route).toBeVisible();
  const confirmedPoints = await route.getAttribute('points');
  await expect(page.getByRole('alert')).toContainText('ĐANG ĐI SAI TUYẾN');
  await expect(route).toHaveAttribute('points', confirmedPoints ?? '');
});

test('F-H reset is cancelled without a request and confirmed exactly once', async ({ page }) => {
  let resetCalls = 0;
  await page.route('**/api/runtime/reset-identities', async (route) => {
    resetCalls += 1;
    await new Promise((resolve) => setTimeout(resolve, 100));
    await route.fulfill({ json: { reset: true, retired_identities: 3, include_sessions: false } });
  });
  await page.goto('/monitor?fixture=driver-isolation');
  await expect(page.getByLabel(/Xe Global ID/)).toHaveCount(3);

  await page.getByRole('button', { name: /Reset Global ID/ }).click();
  await page.getByRole('button', { name: 'Hủy' }).click();
  expect(resetCalls).toBe(0);

  await page.getByRole('button', { name: /Reset Global ID/ }).click();
  await page.getByRole('button', { name: 'Xác nhận reset' }).click();
  await expect.poll(() => resetCalls).toBe(1);
  await expect(page.getByText('Đã reset 3 Global ID.')).toBeVisible();
  await expect(page.getByLabel(/Xe Global ID/)).toHaveCount(0);
});

test('F-I disconnect retains the last snapshot and reports the outage', async ({ page }) => {
  await page.goto('/monitor?fixture=offline');
  await page.waitForTimeout(250);
  await expect(page.getByText(/Mất kết nối Runtime API/)).toBeVisible();
  await expect(page.getByLabel('Xe Global ID 17')).toBeVisible();
});
