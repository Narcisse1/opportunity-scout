// n8n Workflow SDK source for "Opportunity Scout: weekly email".
// Every Monday 07:00: run a live scan through the Scout API, then email the result with Gmail.
// Edit the profile and queries in "Run weekly scan" to change which opportunities are tracked.
import { workflow, node, trigger, ifElse, newCredential, placeholder, expr } from '@n8n/workflow-sdk';

const profile = {
  name: 'Ekang Agritech',
  one_liner: 'Smartphone crop-disease diagnosis for smallholder farmers in Cameroon',
  stage: 'pre-seed',
  sectors: ['agritech', 'AI', 'climate'],
  country: 'Cameroon',
  regions_served: ['Cameroon', 'Central Africa'],
  goals: ['Non-dilutive funding to extend the pilot', 'Cloud and AI API credits'],
  exclude_keywords: ['crypto', 'token sale'],
};

const body = {
  profile,
  queries: [
    'agritech grants Africa 2027 applications open',
    'pre-seed accelerator Africa AI startups 2027',
    'startup competition Cameroon 2027',
  ],
  min_score: 60,
  results_per_query: 4,
  deadline_window_days: 45,
  app_url: 'https://opportunity-scout-98lk.onrender.com',
};

const weekly = trigger({
  type: 'n8n-nodes-base.scheduleTrigger',
  version: 1.2,
  config: {
    name: 'Every Monday 07:00',
    parameters: { rule: { interval: [{ field: 'cronExpression', expression: '0 7 * * 1' }] } },
  },
});

const scan = node({
  type: 'n8n-nodes-base.httpRequest',
  version: 4.2,
  config: {
    name: 'Run weekly scan',
    parameters: {
      method: 'POST',
      url: 'https://opportunity-scout-98lk.onrender.com/api/digest/weekly',
      authentication: 'genericCredentialType',
      genericAuthType: 'httpTemplatedCustomAuth',
      sendBody: true,
      specifyBody: 'json',
      jsonBody: JSON.stringify(body, null, 2),
      options: { timeout: 600000 },
    },
    credentials: { httpTemplatedCustomAuth: newCredential('Scout owner key') },
    retryOnFail: true,
    maxTries: 2,
    waitBetweenTries: 5000,
  },
});

const anything = ifElse({
  version: 2.2,
  config: {
    name: 'Anything to report?',
    parameters: {
      conditions: {
        options: { caseSensitive: true, leftValue: '', typeValidation: 'loose' },
        conditions: [
          {
            leftValue: expr('{{ $json.new_count + $json.upcoming_count }}'),
            operator: { type: 'number', operation: 'gt' },
            rightValue: 0,
          },
        ],
        combinator: 'and',
      },
    },
  },
});

const email = node({
  type: 'n8n-nodes-base.gmail',
  version: 2.1,
  config: {
    name: 'Send weekly email',
    parameters: {
      resource: 'message',
      operation: 'send',
      sendTo: placeholder('Recipient email address(es), comma separated'),
      subject: expr('{{ $json.subject }}'),
      emailType: 'html',
      message: expr('{{ $json.html }}'),
      options: { appendAttribution: false, senderName: 'Opportunity Scout' },
    },
    credentials: { gmailOAuth2: newCredential('Gmail') },
  },
});

export default workflow('opportunity-scout-weekly-email', 'Opportunity Scout: weekly email')
  .add(weekly)
  .to(scan)
  .to(anything.onTrue(email));
